// Minimal Go A2A agent for benchmark testing.
// Uses a2a-go SDK and calls Gemini 3.1 Flash Lite via Vertex AI.
package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"io"
	"strings"

	"cloud.google.com/go/vertexai/genai"
)

func main() {
	project := os.Getenv("GOOGLE_CLOUD_PROJECT")
	if project == "" {
		project = "alanblount-demo"
	}

	ctx := context.Background()
	client, err := genai.NewClient(ctx, project, "global")
	if err != nil {
		log.Fatalf("genai.NewClient: %v", err)
	}
	defer client.Close()

	model := client.GenerativeModel("gemini-3.1-flash-lite")

	// Simple HTTP JSON-RPC handler (minimal A2A server)
	http.HandleFunc("/.well-known/agent-card.json", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"name":"Go Math Agent","version":"1.0.0","description":"Go proof-of-life agent","capabilities":{"streaming":false},"skills":[{"id":"math","name":"Math","description":"Solve math"}],"supportedInterfaces":[{"url":"http://localhost:8766/jsonrpc","protocolBinding":"JSONRPC"}],"defaultInputModes":["text/plain"],"defaultOutputModes":["text/plain"]}`)
	})

	http.HandleFunc("/jsonrpc", func(w http.ResponseWriter, r *http.Request) {
		// Simplified — just extract the text and call Gemini
		
		
		bodyBytes, _ := io.ReadAll(r.Body)
		body := string(bodyBytes)

		// Extract message text (very basic JSON parsing)
		textStart := strings.Index(body, `"text":"`)
		if textStart == -1 {
			textStart = strings.Index(body, `"text": "`)
		}
		userText := "Hello"
		if textStart != -1 {
			rest := body[textStart+8:]
			if strings.HasPrefix(body[textStart:], `"text": "`) {
				rest = body[textStart+9:]
			}
			end := strings.Index(rest, `"`)
			if end > 0 {
				userText = rest[:end]
			}
		}

		resp, err := model.GenerateContent(ctx, genai.Text(userText))
		if err != nil {
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprintf(w, `{"jsonrpc":"2.0","id":1,"error":{"code":-32603,"message":"%s"}}`, err.Error())
			return
		}

		replyText := ""
		if len(resp.Candidates) > 0 && resp.Candidates[0].Content != nil {
			for _, part := range resp.Candidates[0].Content.Parts {
				if t, ok := part.(genai.Text); ok {
					replyText += string(t)
				}
			}
		}

		// Escape for JSON
		replyText = strings.ReplaceAll(replyText, `\`, `\\`)
		replyText = strings.ReplaceAll(replyText, `"`, `\"`)
		replyText = strings.ReplaceAll(replyText, "\n", `\n`)

		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{"jsonrpc":"2.0","id":1,"result":{"kind":"message","role":"agent","parts":[{"kind":"text","text":"%s"}],"messageId":"go-1"}}`, replyText)
	})

	port := os.Getenv("PORT")
	if port == "" {
		port = "8766"
	}
	log.Printf("Go A2A agent listening on :%s", port)
	log.Fatal(http.ListenAndServe(":"+port, nil))
}
