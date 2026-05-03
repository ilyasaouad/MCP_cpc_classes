/**
 * raw-server.ts — DEV / DEBUG ONLY
 *
 * ⚠️  DO NOT USE IN PRODUCTION ⚠️
 *
 * This is a minimal bare-bones Node.js HTTP MCP server used during early
 * development to verify that the MCP SDK transport handshake works before
 * Express was introduced. The `classify_patent` tool here returns a
 * hard-coded placeholder string and does NOT call FastAPI.
 *
 * For the real server, use:
 *   npm run dev   →  src/server/index.ts
 */

import "dotenv/config";
import http from "http";
import { z } from "zod";

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";

const server = new McpServer({
  name: "patent-classifier-mcp-debug",
  version: "0.0.1-debug",
});

server.tool(
  "classify_patent",
  "STUB — returns placeholder. Use index.ts for the real pipeline.",
  { text: z.string() },
  async ({ text }) => {
    console.log("🔥 [DEBUG STUB] TOOL CALLED:", text);
    return {
      content: [{ type: "text", text: "[DEBUG] Hello from raw-server stub — not a real classification" }],
    };
  }
);

const transport = new StreamableHTTPServerTransport({});
await server.connect(transport);

const httpServer = http.createServer(async (req, res) => {
  if (req.method === "POST" && req.url === "/mcp") {
    let body = "";
    req.on("data", (chunk) => (body += chunk));
    req.on("end", async () => {
      try {
        const parsedBody = JSON.parse(body);
        console.log("📥 [DEBUG] Request:", JSON.stringify(parsedBody));
        await transport.handleRequest(req, res, parsedBody);
        console.log("📤 [DEBUG] Response sent");
      } catch (error) {
        console.error("❌ [DEBUG] Error:", error);
        res.writeHead(500, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: String(error) }));
      }
    });
  } else if (req.method === "GET" && req.url === "/") {
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ status: "ok — DEBUG stub server" }));
  } else {
    res.writeHead(404);
    res.end("Not found");
  }
});

const PORT = 3457; // Intentionally different port to avoid collision with index.ts
httpServer.listen(PORT, () => {
  console.log(`🐛 [DEBUG] raw-server running at http://localhost:${PORT}/mcp`);
  console.log("   This is a STUB server. Use index.ts for the real pipeline.");
});
