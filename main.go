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

const defaultDBPath = "agentlab.db"

func main() {
	store, err := openSQLite(defaultDBPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer store.Close()

	ctx := context.Background()
	args := os.Args[1:]
	if len(args) > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, defaultAgentTimeout)
		defer cancel()
	}

	client := &http.Client{Timeout: defaultChatTimeout}
	if err := run(ctx, args, client, defaultOllamaURL, os.Stdin, os.Stdout, os.Stderr, store); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(ctx context.Context, args []string, client *http.Client, baseURL string, stdin io.Reader, stdout, stderr io.Writer, store ConversationStore) error {
	a := newAgent(client, baseURL, defaultRegistry())
	if store != nil {
		a.store = store
	}
	a.logger = slog.New(slog.NewJSONHandler(stderr, nil))

	if len(args) == 0 {
		return runREPL(ctx, a, stdin, stdout)
	}

	id, err := a.store.New(ctx)
	if err != nil {
		return err
	}
	reply, err := a.run(ctx, id, strings.Join(args, " "))
	if err != nil {
		return err
	}

	_, err = fmt.Fprintln(stdout, reply)
	return err
}
