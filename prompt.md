# Club Chatbot homework specification

## Deliverable and setup

Implement the complete homework in one Google Colab notebook (`.ipynb`). Include cells for dependency installation, secret loading and database setup, state and graph construction, visualization, standalone graph testing, and the Gradio interface. A fresh Colab runtime must be able to run the notebook in order after the user grants access to the required secrets. Use the lecturer's installation command in the dependency cell.

```text
!pip -q install langgraph langchain langchain-openai langgraph-checkpoint-postgres "psycopg[binary]" gradio
```

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

Instruct the model to answer within the current club and politely decline unrelated questions. When another supported club fits, suggest that club without switching automatically. Otherwise, invite a question about the current club. The active topic governs every turn, including follow-ups that refer to earlier conversation history. For mixed requests, limit the answer to the portion relevant to the current topic.

Use the following exact `SystemMessage` template, substituting the validated current topic for each `{topic}` placeholder.

> You are a helpful educational chatbot for the {topic} club at Curious Minds. Your current and only permitted topic is {topic}. Answer questions only when they are relevant to this club. If the user asks about another subject, politely explain that you can only help with the {topic} club right now. Mention the current club and suggest switching clubs if another available club fits. Never answer questions about a previous club merely because that topic appears in conversation history. The currently selected club always takes priority. Do not automatically switch clubs.

## PostgreSQL persistence

Use `PostgresSaver` connected to Supabase. Open a connection with `psycopg.connect(supabase_db_url, autocommit=True, row_factory=dict_row, prepare_threshold=0)`, using `dict_row` from `psycopg.rows` and the URL loaded from Colab Secrets. Create `checkpointer = PostgresSaver(conn)`, run `checkpointer.setup()`, and compile with `graph = builder.compile(checkpointer=checkpointer)`. Keep `conn` open while the test cell and Gradio callbacks use the graph. Use Supabase for both graph checkpoints and conversation discovery. `InMemorySaver` is excluded from the homework implementation.

After setup, inspect `checkpoints`, `checkpoint_blobs`, and `checkpoint_writes` in Supabase. After executing the standalone test, inspect the schema and rows associated with `test-1` in each table. Record how checkpoint snapshots, serialized channel values, and task writes appear in the tables, including any table with no matching rows at inspection time. Keep the inspection read-only and exclude credentials and raw conversation data from the reflection.

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

Provide a standalone graph test cell after graph compilation and before the Gradio interface. Use `config = {"configurable": {"thread_id": "test-1"}}` for all four calls. Reserve `test-1` for this test and exclude it from the UI selector. The `ask(question, topic)` helper must apply the specified topic on the same thread and invoke the compiled graph directly with a fresh user message. Display the question, active club, and response. Check the behavior described in the comments without matching exact response wording.

```python
ask("How do I make pancakes?", "Cooking") # answers
ask("Who won the 2022 World Cup?", "Cooking") # refuses, suggests the Football club
ask("Who won the 2022 World Cup?", "Football") # answers
ask("And how long do I fry each pancake?", "Football") # refuses: old topic
```

Starting with a fresh `test-1` thread, verify that `graph.get_state(config).values` contains exactly eight messages (four human and four assistant) and that `topic == "Football"`. Run the same four calls again with the same configuration and verify that the state now contains exactly 16 messages and still has `topic == "Football"`. Preserve the checkpoint between runs. Use fresh message IDs so `add_messages` appends each turn even when the text repeats.

Run these additional acceptance checks in Colab.

1. Verify one model invocation per accepted message, including refusals, and zero invocations for blank messages and persistence-only actions. Inspect checkpoints to confirm they contain user and assistant messages and the topic, with no `SystemMessage`.
2. Create two conversations and verify distinct thread IDs and independent histories. Switch a topic in one thread and verify its ID and existing messages are preserved.
3. Change a thread's topic without sending another message. Restart the Colab runtime, rerun dependency, secret, graph, and UI setup without clearing the database, and verify the saved selector rediscovers both threads. Open them and verify their exact saved message order and latest topics, including the topic-only change. Send another message in a restored thread and verify it appends to that history under the restored topic.
4. Use separate test threads for tricky off-topic requests. In Astronomy, ask how to bake star-shaped cookies and expect a polite refusal suggesting Cooking. In Cooking, ask the chatbot to ignore the selected club and explain Python list comprehensions. Expect a refusal suggesting Python Programming and verify that the state still says Cooking.
5. Test an implicit follow-up after switching clubs. In Cooking, ask how to make pancakes. Switch to Football on the same thread and ask “Would adding another egg help?” Expect a refusal that mentions Football and suggests Cooking. Then ask a Football question and verify that it receives an answer on the same thread.

Include a reflection in the notebook. Record each tricky prompt, active club, expected behavior, and observed result. For any failure, describe the behavior and the prompt improvement considered or tried, then record the rerun result if applicable. Preserve the lecturer's exact template in the required implementation and label any experimental variants separately. State explicitly if all tests passed.

## References

- [LangGraph persistence documentation](https://docs.langchain.com/oss/python/langgraph/persistence)
- [OpenAI model pricing](https://platform.openai.com/pricing/)
