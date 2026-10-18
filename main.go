package main

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"strings"
)

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), defaultAgentTimeout)
	defer cancel()

	client := &http.Client{Timeout: defaultChatTimeout}
	if err := run(ctx, os.Args[1:], client, defaultOllamaURL, os.Stdout, os.Stderr); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(ctx context.Context, args []string, client *http.Client, baseURL string, stdout, stderr io.Writer) error {
	if len(args) < 1 {
		return fmt.Errorf("usage: go run . <instruction>")
	}

	instruction := strings.Join(args, " ")
	a := newAgent(client, baseURL, defaultRegistry())
	a.logger = slog.New(slog.NewJSONHandler(stderr, nil))

	reply, err := a.run(ctx, instruction)
	if err != nil {
		return err
	}

	_, err = fmt.Fprintln(stdout, reply)
	return err
}
