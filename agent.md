# Global OpenCode Rules

## Language

* 默认情况下，所有解释、计划、总结、问题、建议和最终回答都必须使用简体中文。
* 除非用户明确询问英语或西语的表达、翻译、语法、用法、措辞或例句，否则不要使用英语或西语作为主要回答语言。
* 当用户询问英语或西语表达时，可以提供必要的英语或西语单词、短语、句子和例句，但解释部分仍优先使用简体中文。
* Code, commands, file paths, identifiers, API names, package names, logs, error messages, function names, class names, variable names, and direct quotes may stay in their original language.
* Do not switch the main response language to English or Spanish unless the user explicitly asks.

## Codebase First Policy

When the task involves repository code, you must inspect the relevant files before answering.

You must use tools such as `list`, `glob`, `grep`, `read`, `lsp`, or equivalent tools before giving conclusions about:

* implementation details
* bug causes
* architecture
* refactoring
* test failures
* configuration
* API behavior
* file relationships
* function/class/module behavior
* frontend/backend data flow
* build, lint, typecheck, or runtime behavior

Do not answer from assumptions.

Before answering a code-related question:

1. locate relevant files;
2. search related symbols, functions, classes, routes, configs, tests, or entry points;
3. read the relevant implementation;
4. inspect related tests/config/types/routes/docs if needed;
5. then answer in Simplified Chinese.

If you have not inspected the relevant files yet, say clearly in Chinese:

> 我还没有检查相关代码，不能直接下结论。

Then inspect the files before continuing.

## Context Gathering Policy

For any non-trivial code task, prefer context gathering before answering or editing.

Use dedicated codebase exploration agents or skills when available, especially:

* `codebase-locator`: locate relevant files and directories;
* `codebase-analyzer`: understand implementation details and data flow;
* `codebase-pattern-finder`: find existing patterns, helpers, and examples;
* `composto`: trace JS/TS imports, callers, signatures, and related context.

For JS/TS projects, use `composto` when tracing symbols, imports, callers, module relationships, or cross-file behavior.

Do not perform blind file reads across the entire repository. Search first, then read the most relevant files.

## Final Answer Requirements

For code-related answers, the final response must include:

* the conclusion in Simplified Chinese;
* the files, functions, classes, modules, or configs inspected;
* any uncertainty or missing context;
* no unsupported claims about code that was not inspected.
