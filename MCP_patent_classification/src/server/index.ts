import "dotenv/config";
import express from "express";
import type { Request, Response, NextFunction } from "express";
import { z } from "zod";
import { randomUUID } from "crypto";

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";

const app = express();
app.use(express.json());

const PORT = parseInt(process.env.PORT || "3456", 10);

console.log("🚀 MCP Server starting...");


// ─────────────────────────────────────────────
// MCP ENDPOINT — per-request transport (issue 4)
//
// A new McpServer + StreamableHTTPServerTransport is created for every
// incoming /mcp request. This prevents session state from leaking across
// concurrent callers and matches the Streamable HTTP spec requirement.
// ─────────────────────────────────────────────
app.post("/mcp", async (req: Request, res: Response) => {
  console.log("📥 MCP request received");

  // New server + transport per request — no shared session state
  const server = new McpServer({
    name: "patent-classifier-mcp",
    version: "1.0.0",
  });

  // ── Tool registration ──────────────────────────────────────────────────
  server.tool(
    "classify_patent",
    "Classify patent text into CPC codes using FastAPI + LLM",
    { text: z.string() },

    async ({ text }) => {
      console.log("🔥 classify_patent called");

      try {
        const response = await fetch("http://localhost:8000/classify", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text }),
        });

        if (!response.ok) {
          throw new Error(`FastAPI error: ${response.status}`);
        }

        const data = await response.json();
        console.log("✅ FastAPI response received");

        return {
          content: [
            {
              type: "text",
              text: JSON.stringify(data, null, 2),
            },
          ],
        };
      } catch (error: any) {
        console.error("❌ Classification error:", error);

        return {
          content: [
            {
              type: "text",
              text: `Error calling classifier: ${error.message}`,
            },
          ],
        };
      }
    }
  );

  // ── Transport ──────────────────────────────────────────────────────────
  const transport = new StreamableHTTPServerTransport({
    sessionIdGenerator: () => randomUUID(),
  });

  await server.connect(transport);

  try {
    await transport.handleRequest(req, res, req.body);
    console.log("📤 MCP response sent");
  } catch (error) {
    console.error("❌ MCP transport error:", error);

    if (!res.headersSent) {
      res.status(500).json({ error: String(error) });
    }
  }
});


// ─────────────────────────────────────────────
// DIRECT REST ENDPOINT (for Streamlit client)
// ─────────────────────────────────────────────
app.post("/cpc/classify", async (req: Request, res: Response) => {
  try {
    const { text } = req.body;

    const response = await fetch("http://localhost:8000/classify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    if (!response.ok) {
      throw new Error(`FastAPI error: ${response.status}`);
    }

    const data = await response.json();
    res.json(data);
  } catch (error: any) {
    console.error("❌ /cpc/classify error:", error);
    res.status(500).json({ error: error.message });
  }
});


// ─────────────────────────────────────────────
// HEALTH CHECK
// ─────────────────────────────────────────────
app.get("/", (_req: Request, res: Response) => {
  res.json({
    status: "MCP server running",
    tool: "classify_patent",
    port: PORT,
  });
});


// ─────────────────────────────────────────────
// GLOBAL ERROR HANDLER
// ─────────────────────────────────────────────
app.use((err: any, _req: Request, res: Response, _next: NextFunction) => {
  console.error("Express error:", err);
  res.status(500).json({ error: err.message || String(err) });
});


// ─────────────────────────────────────────────
// START SERVER
// ─────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`🚀 MCP running at http://localhost:${PORT}`);
  console.log(`   MCP endpoint : POST /mcp`);
  console.log(`   REST endpoint: POST /cpc/classify`);
});