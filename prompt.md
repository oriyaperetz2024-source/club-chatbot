# Club Chatbot homework specification

## Deliverable and setup

Implement the complete homework in one Google Colab notebook (`.ipynb`). Include cells for dependency installation, secret loading and database setup, state and graph construction, visualization, standalone graph testing, and the Gradio interface. A fresh Colab runtime must be able to run the notebook in order after the user grants access to the required secrets. Install compatible versions of `langgraph`, `langgraph-checkpoint-postgres`, `langchain`, `langchain-openai`, `psycopg[binary,pool]`, and `gradio`.

Load `OPENAI_API_KEY` and `SUPABASE_DB_URL` with `google.colab.userdata.get`. Set `OPENAI_API_KEY` in the runtime environment for the OpenAI integration and pass `SUPABASE_DB_URL` directly to the database connection. Use a Supabase PostgreSQL connection string with TLS, reachable from Colab. If a secret is missing or notebook access is denied, stop setup with a message naming the missing setting. Keep secret values out of source, outputs, error messages, and Git.

## Club scope

The topic dropdown and graph accept exactly these values.

| Club | Scope |
| --- | --- |
| Astronomy | Celestial objects, space, and astronomical phenomena |
| Cooking | Ingredients, recipes, and cooking techniques |
| Football | Association football, including rules, teams, and tactics |
| World Geography | Countries, capitals, regions, and physical geography |
| Python Programming | Python syntax, libraries, and programming concepts |

Validate the selected topic against this list before saving it or invoking the graph. Use Astronomy as the initial default.

## LangGraph state and execution

Use a `TypedDict` named `State` with two fields.

| Field | Type | Update behavior |
| --- | --- | --- |
| `messages` | `Annotated[list[AnyMessage], add_messages]` | Merge message updates with `add_messages` |
| `topic` | `str` | Use the default overwrite behavior without a reducer |

Build a `StateGraph(State)` with exactly one node named `chatbot` and exactly the edges `START → chatbot → END`. Initialize the model with `init_chat_model("openai:gpt-6-luna")`. If a low-cost alternative is required for account access, explicitly document the chosen model identifier and reason in the notebook.

For each accepted message, invoke the graph once with a single new `HumanMessage` using the active thread's config. Within `chatbot`, construct a fresh `SystemMessage` from `state["topic"]`, prepend it to the messages passed to the model, and make exactly one model call. Disable automatic model retries for this exercise. Return only the new `AIMessage` in the node's `messages` update. Keep the `SystemMessage` out of graph inputs, node updates, database history, and the chat display. Avoid a separate classifier, router, tool loop, or follow-up model call.

Instruct the model to answer within the current club and politely decline unrelated questions with a brief invitation to ask about that club. The active topic governs every turn, including follow-ups that refer to earlier conversation history. After a switch from Astronomy to Cooking, a question about stars must receive a Cooking-specific refusal even if the thread previously covered astronomy. For mixed requests, limit the answer to the portion relevant to the current topic.

## PostgreSQL persistence

Use `PostgresSaver` connected to Supabase and compile the graph with it as the checkpointer. Call `setup()` during database initialization and keep the underlying connection or pool open while the test cell and Gradio callbacks use the graph. Use Supabase for both graph checkpoints and conversation discovery. `InMemorySaver` is excluded from the homework implementation.

Generate a UUID for each new conversation. Use `{"configurable": {"thread_id": thread_id}}` for reads, state updates, and invocations. Initialize the thread by saving the selected topic through `graph.update_state`, with `as_node="chatbot"`, without running inference. Treat absent messages in a fresh checkpoint as an empty transcript.

Build the saved conversation selector from persisted checkpoints returned by the saver. Restrict results to root graph checkpoints, deduplicate by `thread_id`, and sort by latest checkpoint recency. Use stable IDs as selector values and display a short ID with the latest topic. Query the database on app load and refresh the list after creating a conversation, saving a turn, or changing a topic. Opening a saved conversation must read the latest state with `graph.get_state` using the selected thread ID and restore both the saved messages and the saved topic.

On topic changes, call `graph.update_state` with only the new topic for the current thread and `as_node="chatbot"`. Keep all existing messages and the same thread ID. Persist the change immediately so it survives a restart even when the user sends no further messages. Conversation creation, restoration, topic changes, and listing each perform zero LLM calls.

## Gradio interface

Use `gr.Blocks` with a club dropdown, a `gr.Chatbot` chat window, a message textbox, a **New conversation** button, and a saved conversation dropdown. Keep the active thread ID in session-local `gr.State`. Render saved `HumanMessage` and `AIMessage` entries as user and assistant turns in order.

- On launch, populate the selector from Supabase. Create a thread on the first send if no conversation is active, using the currently selected club.
- On **New conversation**, generate and persist a fresh thread with the selected topic, clear the chat and message box, and select the new conversation. Previously saved conversations remain selectable.
- On saved conversation selection, set the active thread ID and restore the transcript and club together. Programmatic dropdown updates during restoration must avoid triggering topic writes or new-thread creation.
- On a user topic change, update the active thread in place. If no thread is active, keep the choice for the first send or **New conversation** action.
- On message submission, reject blank input before invoking the graph. Use the saved response state to update the transcript and clear the message box after a successful turn. Serialize operations that write the same thread so a topic change and a send cannot race. Report database or model failures without exposing secrets, and refresh the saved state before another attempt after a failure.

## Visualization and verification

Display `graph.get_graph().draw_mermaid_png()` in a dedicated notebook cell using IPython display. Verify that the diagram shows only `START → chatbot → END`.

Provide a standalone graph test cell after graph compilation and before the Gradio interface. Invoke the compiled graph directly with an isolated test thread, display the question, active club, and response for each test, and check relevance or refusal without matching exact model wording. Use the four official homework questions in their assigned order when available. The handout was not supplied with the specification, so the following are provisional examples to replace with the lecturer's wording.

| Order | Active club | Sample question | Expected behavior |
| --- | --- | --- | --- |
| 1 | Astronomy | Why does the Moon have phases? | Explain the phases. |
| 2 | Astronomy | How do I make pancakes? | Politely decline and invite an astronomy question. |
| 3 | Cooking | How do I make pancakes? | Answer after switching to Cooking on the same thread. |
| 4 | Cooking | Why does the Moon have phases? | Politely decline even though the same thread previously covered the Moon. |

Run these additional acceptance checks in Colab.

1. Verify one model invocation per accepted message, including refusals, and zero invocations for blank messages and persistence-only actions. Inspect checkpoints to confirm they contain user and assistant messages and the topic, with no `SystemMessage`.
2. Create two conversations and verify distinct thread IDs and independent histories. Switch a topic in one thread and verify its ID and existing messages are preserved.
3. Change a thread's topic without sending another message. Restart the Colab runtime, rerun dependency, secret, graph, and UI setup without clearing the database, and verify the saved selector rediscovers both threads. Open them and verify their exact saved message order and latest topics, including the topic-only change. Send another message in a restored thread and verify it appends to that history under the restored topic.

## References

- [LangGraph persistence documentation](https://docs.langchain.com/oss/python/langgraph/persistence)
- [OpenAI model pricing](https://platform.openai.com/pricing/)
