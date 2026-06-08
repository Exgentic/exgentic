// Package main implements a minimal A2A math agent in Go.
//
// It uses the official a2a-go SDK (github.com/a2aproject/a2a-go) to serve
// an A2A endpoint that answers math questions using simple arithmetic.
//
// Usage:
//
//	go run main.go --port 9200
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	"regexp"
	"strconv"
	"strings"

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/a2asrv"
	"github.com/a2aproject/a2a-go/a2asrv/eventqueue"
)

// mockMathResponse provides basic arithmetic for proof-of-life testing.
func mockMathResponse(text string) string {
	re := regexp.MustCompile(`[-+]?\d*\.?\d+`)
	matches := re.FindAllString(text, -1)
	if len(matches) < 2 {
		return fmt.Sprintf("I received your question: %s", truncate(text, 200))
	}

	var nums []float64
	for _, m := range matches {
		n, err := strconv.ParseFloat(m, 64)
		if err == nil {
			nums = append(nums, n)
		}
	}
	if len(nums) < 2 {
		return fmt.Sprintf("I received your question: %s", truncate(text, 200))
	}

	lower := strings.ToLower(text)
	switch {
	case containsAny(lower, "multiply", "product", "times", "*"):
		result := 1.0
		for _, n := range nums {
			result *= n
		}
		return fmt.Sprintf("%g", result)
	case containsAny(lower, "subtract", "minus", "difference"):
		result := nums[0]
		for _, n := range nums[1:] {
			result -= n
		}
		return fmt.Sprintf("%g", result)
	case containsAny(lower, "divide", "quotient"):
		if nums[1] != 0 {
			return fmt.Sprintf("%g", nums[0]/nums[1])
		}
		return "Cannot divide by zero"
	default:
		// Default: sum all numbers
		var sum float64
		for _, n := range nums {
			sum += n
		}
		return fmt.Sprintf("%g", sum)
	}
}

func containsAny(text string, words ...string) bool {
	for _, w := range words {
		if strings.Contains(text, w) {
			return true
		}
	}
	return false
}

func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

// mathExecutor implements a2asrv.AgentExecutor.
type mathExecutor struct{}

var _ a2asrv.AgentExecutor = (*mathExecutor)(nil)

func (*mathExecutor) Execute(ctx context.Context, reqCtx *a2asrv.RequestContext, q eventqueue.Queue) error {
	// Extract text from the user message
	var userText string
	if reqCtx.Message != nil {
		for _, part := range reqCtx.Message.Parts {
			if tp, ok := part.(a2a.TextPart); ok {
				userText += tp.Text
			}
		}
	}

	if userText == "" {
		response := a2a.NewMessage(a2a.MessageRoleAgent, a2a.TextPart{Text: "No input text received"})
		return q.Write(ctx, response)
	}

	log.Printf("Processing: %s", truncate(userText, 200))
	answer := mockMathResponse(userText)
	log.Printf("Response: %s", truncate(answer, 200))

	response := a2a.NewMessage(a2a.MessageRoleAgent, a2a.TextPart{Text: answer})
	return q.Write(ctx, response)
}

func (*mathExecutor) Cancel(ctx context.Context, reqCtx *a2asrv.RequestContext, q eventqueue.Queue) error {
	return nil
}

func main() {
	port := flag.Int("port", 9200, "Port to listen on")
	host := flag.String("host", "0.0.0.0", "Host to bind to")
	flag.Parse()

	agentCard := &a2a.AgentCard{
		Name:               "Go Math Agent",
		Description:        "A simple Go agent that answers math questions via A2A",
		URL:                fmt.Sprintf("http://%s:%d/a2a", *host, *port),
		Version:            "0.1.0",
		PreferredTransport: a2a.TransportProtocolJSONRPC,
		Capabilities:       a2a.AgentCapabilities{Streaming: true},
		Skills: []a2a.AgentSkill{
			{
				ID:          "math",
				Name:        "Math Problem Solver",
				Description: "Solves math problems and answers questions",
				Tags:        []string{"math", "calculation"},
			},
		},
		DefaultInputModes:  []string{"text/plain"},
		DefaultOutputModes: []string{"text/plain"},
	}

	handler := a2asrv.NewHandler(&mathExecutor{})

	mux := http.NewServeMux()
	mux.Handle("/a2a", a2asrv.NewJSONRPCHandler(handler))
	mux.Handle(a2asrv.WellKnownAgentCardPath, a2asrv.NewStaticAgentCardHandler(agentCard))

	addr := fmt.Sprintf("%s:%d", *host, *port)
	listener, err := net.Listen("tcp", addr)
	if err != nil {
		log.Fatalf("Failed to listen on %s: %v", addr, err)
	}

	log.Printf("Starting A2A Go Math Agent on %s", addr)
	if err := http.Serve(listener, mux); err != nil {
		log.Fatalf("Server error: %v", err)
	}
}
