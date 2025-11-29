from __future__ import annotations

import streamlit as st


def main() -> None:
    st.set_page_config(page_title="Chatbot learnings", page_icon="🧠")
    st.title("Chatbot learnings")
    st.write("Plain words about how we are teaching the chatbot to be kinder.")

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

    st.header("LLM-as-a-Judge learnings")

    st.subheader("1. Rubric anchored judging")
    st.markdown(
        """
LLM-as-a-Judge uses the same five-dimension rubric as the human reviewers, so the scores mean something real.
When Sofia from Cádiz writes about pushing for safe passage, the judge grades how she handled context, tools, and values, not just tone.
We feed the rubric text back to the model in every call, which keeps the reasoning grounded instead of drifting into vague praise.
It also lets us compare runs over time because the judge explains each sub-score, like, "Context quality suffered because no mapping of the militia network was mentioned."
That small note tells the team exactly where to coach the chatbot next.
        """
    )

    st.subheader("2. Scenario diversity injection")
    st.markdown(
        """
To keep the judge honest we randomize the question pool with first-person diary style prompts.
One run may include Omondi in Kisumu asking about barter corridors, while another has Mariana in Recife dealing with rival volunteer groups.
The judge sees typos, informal verbs, and different English levels on purpose, so it learns to score realism instead of polished grammar.
When the chatbot fumbles a detail, like forgetting to acknowledge a mayor in Quezon City, the judge cites that concrete miss in its explanation.
Those grounded notes make the evaluation actionable for both product and training teams.
        """
    )

    st.subheader("3. Why we kept a single judge")
    st.markdown(
        """
We tested a multi-judge setup where three small models voted, but they disagreed wildly on red flags and slowed each batch by 4×.
LLM-as-a-Judge now runs as one strong judge plus deterministic rubric checks, which keeps latency low enough to evaluate 20 questions in a meeting.
When a verdict feels shaky, we re-run just that item with a different temperature and compare the rationales instead of re-judging everything.
This approach gives us transparency similar to a panel without the operational overhead.
It also makes it easy to show an example, like, "Judge flagged Fatou's negotiation in Gao because no follow-up safeguards were proposed," and discuss it with the team the same day.
        """
    )


if __name__ == "__main__":
    main()

