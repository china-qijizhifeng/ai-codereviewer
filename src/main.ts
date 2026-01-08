import { readFileSync } from "fs";
import * as core from "@actions/core";
import OpenAI from "openai";
import { Octokit } from "@octokit/rest";
import parseDiff, { File } from "parse-diff";
import { minimatch } from "minimatch";

const GITHUB_TOKEN: string = process.env.GITHUB_TOKEN || "";
const OPENAI_API_KEY: string = core.getInput("OPENAI_API_KEY");
const OPENAI_API_MODEL: string = core.getInput("OPENAI_API_MODEL");
const OPENAI_BASE_URL: string = core.getInput("OPENAI_BASE_URL");

const octokit = new Octokit({ auth: GITHUB_TOKEN });

const openai = new OpenAI({
  apiKey: OPENAI_API_KEY,
  ...(OPENAI_BASE_URL ? { baseURL: OPENAI_BASE_URL } : {}),
});

interface PRDetails {
  owner: string;
  repo: string;
  pull_number: number;
  title: string;
  description: string;
}

async function getPRDetails(): Promise<PRDetails> {
  const { repository, number } = JSON.parse(
    readFileSync(process.env.GITHUB_EVENT_PATH || "", "utf8"),
  );
  const prResponse = await octokit.pulls.get({
    owner: repository.owner.login,
    repo: repository.name,
    pull_number: number,
  });
  return {
    owner: repository.owner.login,
    repo: repository.name,
    pull_number: number,
    title: prResponse.data.title ?? "",
    description: prResponse.data.body ?? "",
  };
}

async function getDiff(
  owner: string,
  repo: string,
  pull_number: number,
): Promise<string | null> {
  const response = await octokit.pulls.get({
    owner,
    repo,
    pull_number,
    mediaType: { format: "diff" },
  });

  // When mediaType.format is 'diff', Octokit returns raw diff string
  const data = response.data;

  // Runtime type check for safety
  if (typeof data === "string") {
    return data;
  }

  console.error("Unexpected response type from GitHub API");
  return null;
}

async function analyzeCode(
  parsedDiff: File[],
  prDetails: PRDetails,
): Promise<Array<{ body: string; path: string; line: number }>> {
  const comments: Array<{ body: string; path: string; line: number }> = [];

  const CONCURRENCY = 1;
  for (let i = 0; i < parsedDiff.length; i += CONCURRENCY) {
    const batch = parsedDiff.slice(i, i + CONCURRENCY);
    const batchResults = await Promise.all(
      batch.map((file) => analyzeFile(file, prDetails)),
    );

    // Flatten and collect all comments
    for (const fileComments of batchResults) {
      if (fileComments) {
        comments.push(...fileComments);
      }
    }
  }

  return comments;
}

async function analyzeFile(
  file: File,
  prDetails: PRDetails,
): Promise<Array<{ body: string; path: string; line: number }> | null> {
  if (file.to === "/dev/null") return null; // Ignore deleted files

  // Combine all chunks of the file into a single prompt
  const prompt = createPrompt(file, prDetails);
  const aiResponse = await getAIResponse(prompt);

  if (!aiResponse || aiResponse.length === 0) {
    return null;
  }

  return createComment(file, aiResponse);
}

function createPrompt(file: File, prDetails: PRDetails): string {
  // Combine all chunks into a single diff view
  const allChanges = file.chunks
    .map((chunk) => {
      const changes = chunk.changes
        .map((c) => {
          const lineNum =
            "ln" in c && c.ln ? c.ln : "ln2" in c && c.ln2 ? c.ln2 : "";
          return `${lineNum} ${c.content}`;
        })
        .join("\n");
      return `${chunk.content}\n${changes}`;
    })
    .join("\n\n");

  return `You are an experienced code reviewer. Your task is to review the following code changes and provide constructive feedback.

**Instructions:**
- Focus on: bugs, security issues, performance problems, code quality, and best practices
- DO NOT provide positive comments or praise
- DO NOT suggest adding code comments or documentation
- ONLY provide feedback if there are actual issues to address
- Be specific and actionable in your suggestions
- Consider the PR title and description for context

**Required Response Format (JSON):**
{
  "reviews": [
    {
      "lineNumber": <line_number>,
      "reviewComment": "<your review comment in GitHub Markdown format>",
      "severity": "critical" | "warning" | "suggestion"
    }
  ]
}

Severity levels:
- critical: Security vulnerabilities, bugs that will cause failures
- warning: Code quality issues, potential bugs, performance concerns
- suggestion: Style improvements, best practices, minor optimizations

If no issues found, return: {"reviews": []}

**Pull Request Context:**
Title: ${prDetails.title}
Description: ${prDetails.description || "No description provided"}

**File:** ${file.to}

**Code Diff:**
\`\`\`diff
${allChanges}
\`\`\`

Provide your review in valid JSON format.`;
}

interface AIReview {
  lineNumber: string;
  reviewComment: string;
  severity: "critical" | "warning" | "suggestion";
}

async function getAIResponse(prompt: string): Promise<AIReview[] | null> {
  const queryConfig = {
    model: OPENAI_API_MODEL,
    temperature: 0.2,
    top_p: 1,
    frequency_penalty: 0,
    presence_penalty: 0,
  };

  try {
    const request: OpenAI.Chat.Completions.ChatCompletionCreateParamsNonStreaming =
      {
        ...queryConfig,
        response_format: { type: "json_object" },
        messages: [
          {
            role: "user",
            content: prompt,
          },
        ],
      };

    console.log("OpenAI request:", JSON.stringify(request, null, 2));

    const response = await openai.chat.completions.create(request);

    let res = response.choices[0].message?.content?.trim() || "{}";

    const jsonCodeBlockMatch = res.match(/^```(?:json)?\s*\n([\s\S]*?)\n```$/);
    if (jsonCodeBlockMatch) {
      res = jsonCodeBlockMatch[1];
    }

    const parsed = JSON.parse(res);
    return parsed.reviews || [];
  } catch (error) {
    console.error("Error calling OpenAI API:", error);
    if (error instanceof Error) {
      console.error("Error details:", error.message);
    }
    return null;
  }
}

function createComment(
  file: File,
  aiResponses: AIReview[],
): Array<{ body: string; path: string; line: number }> {
  return aiResponses.flatMap((aiResponse) => {
    if (!file.to) {
      return [];
    }

    // Add severity icon to the comment
    const severityIcon = {
      critical: "🔴",
      warning: "⚠️",
      suggestion: "💡",
    }[aiResponse.severity];

    const body = `${severityIcon} **${aiResponse.severity.toUpperCase()}**\n\n${aiResponse.reviewComment}`;

    return {
      body,
      path: file.to,
      line: Number(aiResponse.lineNumber),
    };
  });
}

async function withRetry<T>(
  fn: () => Promise<T>,
  maxRetries = 5,
  baseDelayMs = 2000,
): Promise<T> {
  let lastError: unknown;

  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;
      if (attempt === maxRetries - 1) {
        throw error;
      }

      const isRateLimitError =
        error &&
        typeof error === "object" &&
        "status" in error &&
        error.status === 403 &&
        "message" in error &&
        typeof error.message === "string" &&
        error.message.includes("secondary rate limit");

      if (!isRateLimitError) {
        throw error;
      }

      let delayMs = baseDelayMs * Math.pow(2, attempt);

      if (
        error &&
        typeof error === "object" &&
        "response" in error &&
        error.response &&
        typeof error.response === "object" &&
        "headers" in error.response &&
        error.response.headers &&
        typeof error.response.headers === "object" &&
        "retry-after" in error.response.headers
      ) {
        const retryAfter = Number(error.response.headers["retry-after"]);
        if (!isNaN(retryAfter)) {
          delayMs = Math.max(delayMs, retryAfter * 1000);
          console.log(
            `Using Retry-After header: ${retryAfter}s, waiting ${delayMs}ms`,
          );
        }
      }

      console.log(
        `Rate limit hit, retrying in ${delayMs}ms (attempt ${attempt + 1}/${maxRetries})`,
      );
      await new Promise((resolve) => setTimeout(resolve, delayMs));
    }
  }

  throw lastError;
}

async function createReviewComment(
  owner: string,
  repo: string,
  pull_number: number,
  comments: Array<{ body: string; path: string; line: number }>,
): Promise<void> {
  const MAX_COMMENTS_PER_BATCH = 20;
  const commentsToSend = comments.slice(0, MAX_COMMENTS_PER_BATCH);

  console.log(
    `Creating review with ${comments.length} comments (sending first ${commentsToSend.length}) for ${owner}/${repo}#${pull_number}`,
  );

  await withRetry(() =>
    octokit.pulls.createReview({
      owner,
      repo,
      pull_number,
      comments: commentsToSend,
      event: "COMMENT",
    }),
  );

  console.log(
    `Review created successfully with ${commentsToSend.length} comments`,
  );

  if (comments.length > MAX_COMMENTS_PER_BATCH) {
    const remainingCount = comments.length - MAX_COMMENTS_PER_BATCH;
    console.log(
      `⚠️ ${remainingCount} additional comments not sent (exceeds max batch size of ${MAX_COMMENTS_PER_BATCH})`,
    );
  }
}

async function createReviewSummary(
  owner: string,
  repo: string,
  pull_number: number,
  comments: Array<{ body: string; path: string; line: number }>,
): Promise<void> {
  console.log(`Creating summary comment for ${owner}/${repo}#${pull_number}`);

  const existingComments = await withRetry(() =>
    octokit.issues.listComments({
      owner,
      repo,
      issue_number: pull_number,
    }),
  );

  const botComment = existingComments.data.find(
    (comment) =>
      comment.body?.includes("🤖 AI Code Review Summary") &&
      comment.user?.type === "Bot",
  );

  let summaryBody: string;

  if (comments.length === 0) {
    summaryBody = `## 🤖 AI Code Review Summary\n\n✅ No issues found. The code looks good!`;
  } else {
    const criticalCount = comments.filter((c) => c.body.includes("🔴")).length;
    const warningCount = comments.filter((c) => c.body.includes("⚠️")).length;
    const suggestionCount = comments.filter((c) =>
      c.body.includes("💡"),
    ).length;

    summaryBody = `## 🤖 AI Code Review Summary

**Total Issues Found:** ${comments.length}

 - 🔴 **Critical:** ${criticalCount}
 - ⚠️ **Warnings:** ${warningCount}
 - 💡 **Suggestions:** ${suggestionCount}

**Files Reviewed:** ${new Set(comments.map((c) => c.path)).size}

Please review the inline comments for detailed feedback.`;
  }

  if (botComment) {
    console.log(`Updating existing summary comment (ID: ${botComment.id})`);
    await withRetry(() =>
      octokit.issues.updateComment({
        owner,
        repo,
        comment_id: botComment.id,
        body: summaryBody,
      }),
    );
    console.log(`Summary updated successfully`);
  } else {
    console.log(`Creating new summary comment`);
    await withRetry(() =>
      octokit.issues.createComment({
        owner,
        repo,
        issue_number: pull_number,
        body: summaryBody,
      }),
    );
    console.log(`Summary created successfully`);
  }
}

async function main() {
  const prDetails = await getPRDetails();
  let diff: string | null;
  const eventData = JSON.parse(
    readFileSync(process.env.GITHUB_EVENT_PATH ?? "", "utf8"),
  );

  if (eventData.action === "opened") {
    diff = await getDiff(
      prDetails.owner,
      prDetails.repo,
      prDetails.pull_number,
    );
  } else if (eventData.action === "synchronize") {
    const newBaseSha = eventData.before;
    const newHeadSha = eventData.after;

    const response = await octokit.repos.compareCommits({
      headers: {
        accept: "application/vnd.github.v3.diff",
      },
      owner: prDetails.owner,
      repo: prDetails.repo,
      base: newBaseSha,
      head: newHeadSha,
    });

    diff = String(response.data);
  } else {
    console.log("Unsupported event:", process.env.GITHUB_EVENT_NAME);
    return;
  }

  if (!diff) {
    console.log("No diff found");
    return;
  }

  const parsedDiff = parseDiff(diff);

  const excludePatterns = core
    .getInput("exclude")
    .split(",")
    .map((s) => s.trim());

  const filteredDiff = parsedDiff.filter((file) => {
    return !excludePatterns.some((pattern) =>
      minimatch(file.to ?? "", pattern),
    );
  });

  const comments = await analyzeCode(filteredDiff, prDetails);

  console.log(`Total comments generated: ${comments.length}`);

  if (comments.length > 0) {
    await createReviewComment(
      prDetails.owner,
      prDetails.repo,
      prDetails.pull_number,
      comments,
    );

    await new Promise((resolve) => setTimeout(resolve, 2000));
  }

  await createReviewSummary(
    prDetails.owner,
    prDetails.repo,
    prDetails.pull_number,
    comments,
  );

  console.log("Code review workflow completed");
}

main().catch((error) => {
  console.error("Error:", error);
  process.exit(1);
});
