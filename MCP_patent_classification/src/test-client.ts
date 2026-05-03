async function sendRequest(body: any, sessionId?: string) {
    console.log("\n→ Sending:", JSON.stringify(body));
    
    const headers: Record<string, string> = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream"
    };
    if (sessionId) {
        headers["mcp-session-id"] = sessionId;
    }
    
    const response = await fetch("http://localhost:3456/mcp", {
        method: "POST",
        headers,
        body: JSON.stringify(body)
    });

    console.log("← Status:", response.status, response.statusText);
    console.log("← Headers:", Object.fromEntries(response.headers.entries()));
    
    const text = await response.text();
    console.log("← Raw body:", text || "(empty)");
    
    if (!text) {
        return { response, data: null };
    }
    
    // Parse SSE format: event: message\ndata: {...}\n\n
    const lines = text.split('\n');
    for (const line of lines) {
        if (line.startsWith('data: ')) {
            const jsonStr = line.slice(6);
            try {
                return { response, data: JSON.parse(jsonStr) };
            } catch (e) {
                console.error("Failed to parse SSE data:", jsonStr);
            }
        }
    }
    
    // Fallback: try parsing whole body as JSON
    try {
        return { response, data: JSON.parse(text) };
    } catch (e) {
        return { response, data: text };
    }
}

async function testMCP() {
    let sessionId: string | undefined;

    // 1. Initialize handshake
    const { response: initResponse, data: initResult } = await sendRequest({
        jsonrpc: "2.0",
        id: 1,
        method: "initialize",
        params: {
            protocolVersion: "2024-11-05",
            capabilities: {},
            clientInfo: {
                name: "test-client",
                version: "1.0.0"
            }
        }
    });
    
    sessionId = initResponse.headers.get("mcp-session-id") || undefined;
    console.log("\n✅ Initialize response:", JSON.stringify(initResult, null, 2));
    if (sessionId) console.log("📌 Session ID:", sessionId);

    // 2. Send initialized notification
    await sendRequest({
        jsonrpc: "2.0",
        method: "notifications/initialized"
    }, sessionId);

    // 3. Call tool
    const { data: result } = await sendRequest({
        jsonrpc: "2.0",
        id: 2,
        method: "tools/call",
        params: {
            name: "classify_patent",
            arguments: {
                text: "A neural network system for image recognition"
            }
        }
    }, sessionId);
    
    console.log("\n✅ Tool result:", JSON.stringify(result, null, 2));
}

testMCP().catch(err => {
    console.error("Test failed:", err);
    process.exit(1);
});
