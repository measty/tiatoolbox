import { readFile, writeFile } from "node:fs/promises";

const output = new URL(
  "../../../tiatoolbox/data/visualization/app/index.html",
  import.meta.url,
);
const html = await readFile(output, "utf8");

// Vite preserves the source HTML's CRLF bytes while injecting generated tags
// with LF on Windows. Normalize the committed artifact so builds are identical
// across platforms and do not contain doubled carriage returns.
await writeFile(output, html.replace(/\r\n?/g, "\n"), "utf8");
