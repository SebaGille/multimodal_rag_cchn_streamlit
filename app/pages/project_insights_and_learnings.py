from __future__ import annotations

import streamlit as st


def main() -> None:
    st.set_page_config(page_title="Project insights and learnings", page_icon="🧠")
    st.title("Project insights and learnings")
    st.write("Plain words about how we are teaching the chatbot to be kinder.")

    user_testing_tab, llm_judge_tab, chatbot_tab, agentic_tab = st.tabs(
        ["User testing", "LLM-as-a-Judge", "Chatbot", "Agentic"]
    )

    with user_testing_tab:
        st.header("What went wrong")
        st.markdown(
            """
We built a chatbot that reads the CCHN Field Manual on humanitarian negotiation.
It does great when someone asks a clean textbook question, like "What is the Island of Agreements tool?"
The words in that question match the words in the book, so the chatbot finds an answer fast.

Real life feels messier.
Someone might say, "In Mali, I saw this situation while doing that... what should I have done?"
This kind of question uses places, people, and personal stories.
It almost never copies the exact words from the manual.
The chatbot gets confused, so it sometimes says, "The manual does not give an answer."
That is not true.
The real problem is that the chatbot does not yet know how to connect the messy story to the helpful ideas that already exist in the manual.
            """
        )

        st.header("The Mali example")
        st.markdown(
            """
A humanitarian worker in Mali meets a local commander while trying to open a safe road.
The commander is tense and raises his voice.
The worker answers, but later feels unsure and asks the chatbot, "In Mali, I met this commander who said X and I answered Y, what should I have done?"

The chatbot hunts for the exact words "Mali," "commander," and "what should I have done" in the manual.
It finds almost nothing, so it thinks there is no clear answer.
What it *should* do is look past the country name and the personal story.
It should look for general ideas like how to build trust, how to prepare, and how to handle a difficult counterpart.
Those ideas do live in the manual and could have helped the worker.
            """
        )

        st.header("How we can help the chatbot")

        st.subheader("A. Query rewriting (cleaning the question before search)")
        st.markdown(
            """
**Simple idea:** The chatbot first cleans the messy question and turns it into a clear one that fits the language of the manual.
If someone says, "In Mali I met a commander who yelled at me, what should I have done?", the chatbot rewrites it as, "How should a humanitarian negotiator deal with a hostile counterpart and keep the relationship safe?"
Now the chatbot can search for words like "hostile counterpart," "relationship," and "safety," which appear in the manual.

**Mali example:** With the rewritten question, the chatbot would find the chapters on how to calm the other side and how to stay safe while keeping the talk going.
Instead of saying "no answer," it could point to the steps about preparing, listening, and lowering the heat.
            """
        )

        st.subheader("B. Multi-query retrieval (many small questions instead of one)")
        st.markdown(
            """
**Simple idea:** The chatbot does not search only once.
It breaks the question into several tiny questions, such as "How to deal with a hostile counterpart?", "How to build trust in negotiation?", and "How to prepare for a meeting with an armed actor?"
It searches the manual with all of them.

**Mali example:** These smaller searches give the chatbot more chances to hit the useful pages.
One mini-question might find the trust-building tool, another might find the checklist on preparation, and together they bring back richer advice for the Mali story.
            """
        )

        st.subheader("C. Topic classification (guessing the main theme)")
        st.markdown(
            """
**Simple idea:** Before searching, the chatbot asks itself, "Is this about context, trust, red lines, or something else?"
It sorts the question into a topic bucket.
For the Mali story it might say, "This is about the human element and dealing with a difficult person."
Then it looks harder inside that part of the manual.

**Mali example:** By landing in the "human element" topic, the chatbot jumps straight to tools about reading emotions and caring for the relationship, which is exactly what the worker needed.
            """
        )

        st.subheader("D. Concept hypotheses (what could this question be really about?)")
        st.markdown(
            """
**Simple idea:** The chatbot makes small guesses about the hidden ideas inside the question.
It may guess, "Maybe this is about trust," "Maybe it is about the human element," or "Maybe it is about security and access."
Then it looks for those ideas in the manual.

**Mali example:** These guesses would pull up the trust-building tool, the notes on human elements, and the guidance on creating a safe space.
So the worker would see tips on how to calm the meeting even though the original question never used those exact phrases.
            """
        )

        st.subheader("E. Re-ranking passages (choosing the best pieces of text)")
        st.markdown(
            """
**Simple idea:** The chatbot may find many pieces of text.
It then re-reads them and keeps only the ones that fit the question best, like a teacher saying, "These two pages are the most helpful."

**Mali example:** After re-ranking, the chatbot would keep the passages about hostile counterparts or tense meetings and drop the random parts.
The user would only see the most useful advice for their Mali scene.
            """
        )

        st.subheader("F. Retrieval-aware answer generation (stick to the manual)")
        st.markdown(
            """
**Simple idea:** The chatbot must base every answer on the manual pages it just found.
If the question is vague, it should still say, "Here is how the manual would guide you," instead of, "There is no answer."

**Mali example:** The chatbot could reply, "The manual suggests focusing on the other person's fears, keeping the tone calm, and setting clear steps before asking for access."
It would quote or paraphrase the manual in plain words, so the worker sees practical steps.
            """
        )

        st.subheader("G. Clarifying questions (“Did you mean…?”)")
        st.markdown(
            """
**Simple idea:** When a question is too fuzzy, the chatbot can ask one gentle follow-up, such as, "Are you asking about trust, about your safety, or about what you are allowed to promise?"

**Mali example:** If the worker answers, "I care about staying safe," the chatbot can focus on the safety parts of the manual right away, and the reply becomes more helpful.
            """
        )

        st.subheader("H. Negotiation keyword expander (many ways to say the same thing)")
        st.markdown(
            """
**Simple idea:** People use different words for the same idea.
The chatbot keeps a list of twin phrases.
When someone says "get along with a commander," the chatbot also looks for "build trust" and "keep legitimacy."

**Mali example:** Even if the worker uses everyday words, the chatbot still finds the trust sections and can explain how to rebuild the relationship with the commander.
            """
        )

        st.subheader("I. Section summaries (short maps of each tool)")
        st.markdown(
            """
**Simple idea:** We write short, easy summaries for every big part of the manual.
The chatbot scans these summaries first, like looking at a map before reading a chapter.

**Mali example:** The summary for "Tool 8: Addressing the human elements of the transaction" would pop up.
The chatbot would see that it matches the Mali story and then read the full tool to answer.
            """
        )

        st.subheader("J. Smaller chunks (short pieces of text in the database)")
        st.markdown(
            """
**Simple idea:** We cut the manual into tiny, clear chunks instead of giant pages.
Each chunk covers one strong idea.
When the chatbot searches, it gets sharp, focused bits of advice.

**Mali example:** Instead of dumping a long chapter, the chatbot could surface one small chunk that says, "Before the meeting, list the other person's fears and plan calm answers."
The user gets quick, practical tips that match the Mali scene.
            """
        )

    with llm_judge_tab:
        st.header("LLM-as-a-Judge learnings")

        st.subheader("Rubric-anchored scoring as a measurable model of negotiation competence")
        st.markdown(
            """
Rubric-anchored scoring solidifies negotiation competence into measurable signals.
The judge leans on a five-part rubric (context mapping, mandate clarity, principled reasoning, etc.) that turns the field manual’s abstractions into explicit variables.
Because the rubric text accompanies every evaluation call, the judge behaves like a fixed scoring function: identical inputs go through identical criteria, yielding comparable outputs.
Each sub-score explanation works as the qualitative evidence for the numeric label, so when the judge highlights something like “external factors identified but not linked to strategy,” the team immediately knows which dimension slipped and how to improve it.
            """
        )

        st.subheader("Scenario diversity as robustness testing")
        st.markdown(
            """
Scenario diversity functions as robustness testing.
Prompts span messy phrasing, typos, and multiple English levels to ensure the scorer survives noisy, real-world inputs rather than overfitting to tidy scripts.
The judge’s critiques—say, noting that a local stakeholder from the prompt never reappeared in the answer—double as actionable error signals, guiding systematic fixes instead of cosmetic style tweaks.
            """
        )

        st.subheader("Single deterministic judge as a controlled scoring baseline")
        st.markdown(
            """
Running a single deterministic judge establishes a controlled baseline.
With a locked-in evaluator, any score drift stems from the chatbot, not the rubric, enabling clean comparisons across batches.
When a verdict feels borderline, the team can re-run that exact case at a different temperature without re-processing the entire dataset.
This keeps latency and cost predictable while paving the way for future experimental sophistication: multi-judge ensembles, inter-rater agreement tracking, and even confidence intervals once the baseline is fully understood.
            """
        )

    with chatbot_tab:
        st.header("Live conversation learning")
        st.markdown(
            """
We also tried to hold a live back-and-forth conversation using the same single-turn retrieval stack.
It felt flat, because that stack is only built to answer one question grounded in a few chunks, not to respond like a partner in dialogue.
Here is what we saw in that real chat:

1. **No turn-taking.** When the user added the short update "food might be perished," the chatbot ignored it and launched a fresh essay, so the flow broke immediately.
2. **Copy-paste answers.** The follow-up reply reused the same wall of text, citations, and boilerplate closing, so it looked like a template instead of a thought that reacts.
3. **No compression.** Later turns stayed as long and formal as the first answer, even though the user only typed a fragment, so the pacing never adjusted.
4. **Zero conversational markers.** There was no "got it," no clarification, no reformulation—nothing that signals the model heard the new cue.
5. **Context reset.** The assistant re-explained legitimacy, trust-building, and sourcing from the manual even though that context was already established, which made the user feel unheard.
6. **Mechanical follow-up question.** Both turns ended with nearly the same question about local authorities, which exposed that the model is running a pattern, not listening.
7. **No use of the new fact.** The concern about spoiled food never shaped the advice, so the practical barrier (food safety) stayed unaddressed.
8. **No negotiation arc.** A human negotiator would shift into diagnosing the barrier, probing the concern, and proposing next steps; the chatbot just repeated generic guidance.

Conclusion: when we reuse a one-shot, grounded-answer feature for multi-turn dialogue, we create a chatbot that talks *at* people instead of *with* them.
The fix is not only better retrieval, but also stateful turn memory, answer compression, and explicit negotiation reasoning so each reply feels like part of the same conversation.
            """
        )

    with agentic_tab:
        st.header("Agentic concepts under review")
        st.markdown(
            """
We have not switched these behaviors on yet; they remain design explorations.
Each idea below includes the upside we expect for frontline negotiators and the risk we must manage before any live launch.
            """
        )

        st.subheader("A. Dynamic tool selection (pick the right muscles)")
        st.markdown(
            """
**Potential benefit:** The agent could decide when to search, cite, or run a quick calculation so simple reminders arrive faster while complex dilemmas still get the full retrieval treatment.
**Risk to watch:** Latency and cost can spike if the model keeps spawning extra tool calls. We would need strict limits and clear UX signals whenever an answer takes longer.
            """
        )

        st.subheader("B. Multi-step planning (plan → act → revise)")
        st.markdown(
            """
**Potential benefit:** Asking the agent to outline its steps (rewrite, retrieve, critique, finalize) should surface traceable reasoning that product managers can audit.
**Risk to watch:** Every extra step is another LLM call. Without a cap, the experience could feel sluggish or expensive, so we only consider it inside a tight step budget.
            """
        )

        st.subheader("C. External action hooks (help people move work forward)")
        st.markdown(
            """
**Potential benefit:** Allowing the agent to log a follow-up task or ping a negotiation knowledge base would bridge “advice” and “action,” making the chatbot feel like a teammate.
**Risk to watch:** Each hook touches governance, data retention, and human approvals. We need auditable logs and manual overrides before connecting to any real system.
            """
        )

        st.subheader("D. Self-reflection loop (let the agent critique itself)")
        st.markdown(
            """
**Potential benefit:** A built-in critique pass that checks citations or requests more evidence should reduce hallucinations and produce clearer rationales for PM reviews.
**Risk to watch:** Reflection adds another round-trip to the LLM. We must ensure the quality gain outweighs the latency hit, especially for short questions.
            """
        )

        st.subheader("E. Lightweight memory (remember patterns, not people)")
        st.markdown(
            """
**Potential benefit:** Storing short summaries of recent evaluations (e.g., “struggled with access & security cases”) could help the agent adjust retrieval emphasis without human nudging.
**Risk to watch:** Any memory layer risks leaking session details. We must keep the notes aggregate and forgetful so no personal data or sensitive scenario specifics persist.
            """
        )

        st.subheader("How we would test agentic mode safely")
        st.markdown(
            """
- We would use a **three-step harness** (plan, act, reflect). If the agent needs more, it would fall back to the standard grounded answer so the experience stays predictable.
- We would start on **sandboxed dilemmas** before touching live chats, letting us replay every tool call and compare scores with the deterministic judge.
- The **tool palette would stay narrow** (search, read, cite). External integrations remain behind feature flags until the governance story is ready.
- Agentic mode would ship behind an **opt-in toggle** so evaluators can compare versions side by side and switch it off instantly if something feels off.
- Each run would emit **full telemetry**—plans, retrieved chunks, critiques—so product and safety can trace odd behavior and roll back in minutes.
            """
        )


if __name__ == "__main__":
    main()
