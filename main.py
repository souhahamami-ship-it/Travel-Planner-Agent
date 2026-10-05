from agent import new_conversation, run_agent


def main():
    messages = new_conversation()
    print("Travel assistant ready. Type 'exit' to quit.")

    while True:
        try:
            user_request = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if user_request.lower() in {"exit", "quit"}:
            break
        if not user_request:
            continue

        answer = run_agent(user_request, messages)
        print(f"\nTravel Assistant:\n{answer}")


if __name__ == "__main__":
    main()