import os
import re
from typing import Optional
from dotenv import load_dotenv

import streamlit as st

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

load_dotenv()

embeddings = OllamaEmbeddings(
    model=os.getenv("EMBEDDING_MODEL"),
)

vector_store = Chroma(
    collection_name=os.getenv("COLLECTION_NAME"),
    embedding_function=embeddings,
    persist_directory=os.getenv("DATABASE_LOCATION"),
)

llm = init_chat_model(
    os.getenv("CHAT_MODEL"),
    model_provider=os.getenv("MODEL_PROVIDER"),
    temperature=0.3,
)


def retrieve(query: str) -> str:
    retrieved_docs = vector_store.similarity_search(query, k=3)
    serialized = ""
    for doc in retrieved_docs:
        serialized += f"Source: {doc.metadata['source']}\nContent: {doc.page_content}\n\n"
    return serialized


_QUIZ_TAG_RE = re.compile(
    r"\[\[QUIZ_RESULT:\s*(CORRECT|WRONG|SKIP)\s*\]\]",
    re.IGNORECASE,
)

def extract_quiz_tag(text: str) -> tuple:
    """Returns (tag or None, cleaned text with all tags removed)."""
    matches = list(_QUIZ_TAG_RE.finditer(text))
    tag = matches[-1].group(1).upper() if matches else None
    cleaned = _QUIZ_TAG_RE.sub("", text).strip()
    return tag, cleaned


def heuristic_result(ai_text: str) -> Optional[str]:
    """
    Fallback when model omits the tag.
    Returns 'correct', 'wrong', or None.
    Heuristic checks wrong FIRST — small models praise even wrong answers,
    so wrong markers are treated as stronger signal.
    """
    lower = ai_text.lower()

    wrong_markers = [
        "incorrect", "not quite", "not correct", "that's not right",
        "that is not right", "unfortunately", "wrong answer",
        "not the right answer", "is not right", "is wrong",
        "not accurate", "not exactly", "close, but",
    ]
    if any(m in lower for m in wrong_markers):
        return "wrong"

    correct_markers = [
        "correct!", "that's correct", "that is correct", "you're correct",
        "you are correct", "well done", "great job", "excellent!",
        "spot on", "nailed it", "perfect!", "good answer", "absolutely right",
        "congratulations", "great answer", "well answered",
    ]
    if any(m in lower for m in correct_markers):
        return "correct"

    return None


# Do not trust CORRECT/WRONG on empty / near-empty model output (common with small models).
_MIN_VISIBLE_CHARS_FOR_SCORE = 10

_GARBAGE_USER_REPLIES = frozenset(
    {"?", "??", "???", "yes?", "no?", "ok", "okay", "k", "idk", "...", "..", "hmm", "hm"}
)


def _looks_like_meaningful_answer(user_text: str) -> bool:
    s = user_text.strip().lower()
    if not s or s in _GARBAGE_USER_REPLIES:
        return False
    if len(s) <= 2 and s.strip("?") == "":
        return False
    return True


def apply_outcome(
    tag: Optional[str],
    fallback: Optional[str],
    user_input: str,
    visible_assistant_text: str,
) -> bool:
    """Update score at most once. Returns True if a CORRECT/WRONG was applied."""
    if not _looks_like_meaningful_answer(user_input):
        return False

    visible = visible_assistant_text.strip()
    if len(visible) < _MIN_VISIBLE_CHARS_FOR_SCORE:
        return False

    outcome = None

    if tag in ("CORRECT", "WRONG"):
     outcome = tag.lower()
    h = heuristic_result(visible_assistant_text)
    if h and h != outcome:
        outcome = h  # heuristic overrides only if it disagrees
    elif tag == "SKIP":
      return False
    elif tag is None:
     outcome = fallback  # no tag, use fallback
 
    if outcome == "correct":
        st.session_state.score += 1
        st.session_state.total_questions += 1
        return True
    if outcome == "wrong":
        st.session_state.total_questions += 1
        if st.session_state.last_question_topic:
            st.session_state.weak_areas.append(st.session_state.last_question_topic)
        return True
    return False


def _assistant_asks_new_question(ai_text: str) -> bool:
    """True if this turn looks like a new quiz question (not only a one-line ack)."""
    t = ai_text.strip()
    if "?" not in t or len(t) < 50:
        return False
    lower = t.lower()
    if any(x in lower for x in ("choose", "select one", "which of the following", "a)", "b)", "c)", "d)")):
        return True
    return t.count("?") >= 1 and len(t) > 120


def extract_question_topic(ai_text: str) -> Optional[str]:
    """Detect topic keyword from AI response when it asks a new question."""
    if not _assistant_asks_new_question(ai_text):
        return None
    lower = ai_text.lower()
    topics = [
        "python", "machine learning", "deep learning", "neural network",
        "langchain", "llm", "large language model", "numpy", "pandas",
        "scikit", "transformer", "lstm", "gru", "attention", "nlp", "rag",
        "vector database", "variable", "function", "loop", "class", "object",
        "list", "dictionary", "tuple", "inheritance", "recursion", "algorithm",
    ]
    for topic in topics:
        if topic in lower:
            return topic.title()
    return "Quiz"


def build_system_prompt(level: str, score: int, total: int, weak_areas: list,
                         user_input: str, chat_history: str, context: str) -> str:
    weak_str = ", ".join(weak_areas) if weak_areas else "none identified yet"
    accuracy = round((score / total) * 100) if total > 0 else 0

    if total == 0:
        difficulty = "Start with foundational questions to assess the user's level."
    elif accuracy >= 75:
        difficulty = "The user is doing well. Increase difficulty with deeper conceptual questions."
    elif accuracy >= 40:
        difficulty = "Mix easy and hard questions."
    else:
        difficulty = "The user is struggling. Explain concepts more clearly before quizzing."

    level_instruction = (
        "Use simple language, avoid jargon, and give relatable analogies."
        if level == "Beginner"
        else "Use technical language, go deep into concepts, challenge the user."
    )

    return f"""You are an expert AI Tutor specializing in Python and AI programming.
Your job is to QUIZ and TEST the user, not just answer questions.

STUDENT LEVEL: {level}
CURRENT SCORE: {score}/{total} ({accuracy}%)
WEAK AREAS: {weak_str}

LEVEL INSTRUCTION: {level_instruction}
DIFFICULTY: {difficulty}

RETRIEVED CONTEXT (use this to ground your answer and cite sources):
{context}

HOW TO BEHAVE:
1. Ground all answers in the RETRIEVED CONTEXT above. Cite sources as: Source: <url>
2. Ask quiz questions — multiple choice (A/B/C/D) or open-ended.
3. When the user answers, evaluate it clearly, explain why, then ask the next question.
4. Use plain encouraging language. Do NOT state running scores — that is in the sidebar.
5. MANDATORY LAST LINE: End every reply with EXACTLY one of these tags on its own line:
   - [[QUIZ_RESULT:CORRECT]]  — you graded the user's answer as correct
   - [[QUIZ_RESULT:WRONG]]    — you graded the user's answer as wrong
   - [[QUIZ_RESULT:SKIP]]     — no answer was graded (greeting, topic request, first question, or user message was not a real answer)
   The tag is stripped in the UI. Always write a full helpful message BEFORE the tag (at least a few sentences when grading).

CHAT HISTORY:
{chat_history}

USER INPUT:
{user_input}"""

st.set_page_config(page_title="Python & AI Tutor", page_icon="🎓")
st.title("🎓 Python & AI Programming Tutor")

# ── Session state ──────────────────────────────────────────────────────────────
for key, default in {
    "messages": [],
    "score": 0,
    "total_questions": 0,
    "weak_areas": [],
    "level": "Beginner",
    "last_question_topic": "",
    "awaiting_answer": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    selected_level = st.selectbox(
        "Your Level",
        ["Beginner", "Advanced"],
        index=0 if st.session_state.level == "Beginner" else 1,
    )
    if selected_level != st.session_state.level:
        st.session_state.level = selected_level

    if st.button("🔄 Reset Session"):
        for key, default in {
            "messages": [],
            "score": 0,
            "total_questions": 0,
            "weak_areas": [],
            "last_question_topic": "",
            "awaiting_answer": False,
        }.items():
            st.session_state[key] = default
        st.rerun()

    st.divider()
    st.header("Your Progress")

    total = st.session_state.total_questions
    score = st.session_state.score
    accuracy = round((score / total) * 100) if total > 0 else 0

    col1, col2 = st.columns(2)
    col1.metric("Correct", score)
    col2.metric("Wrong", total - score)

    if total > 0:
        st.progress(accuracy / 100, text=f"Accuracy: {accuracy}%")
    else:
        st.info("No graded answers yet.")

    if st.session_state.weak_areas:
        st.divider()
        st.header("⚠️ Weak Areas")
        for area in set(st.session_state.weak_areas):
            st.warning(area)

for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

user_input = st.chat_input("Type your answer, ask a topic, or say 'start quiz'...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append(HumanMessage(content=user_input))

    chat_history_text = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in st.session_state.messages[:-1]  # exclude current input
    )

    context = retrieve(user_input)

    system_prompt = build_system_prompt(
        level=st.session_state.level,
        score=st.session_state.score,
        total=st.session_state.total_questions,
        weak_areas=st.session_state.weak_areas,
        user_input=user_input,
        chat_history=chat_history_text,
        context=context,
    )

    with st.spinner("Thinking..."):
        result = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                *st.session_state.messages,
            ]
        )

    raw = result.content if hasattr(result, "content") else str(result)

    tag, cleaned_visible = extract_quiz_tag(raw)
    body_ok = bool(cleaned_visible.strip()) and len(cleaned_visible.strip()) >= _MIN_VISIBLE_CHARS_FOR_SCORE

    ai_display = cleaned_visible.strip() or (
        "_(No response generated — please try again or reset the session.)_"
    )

    scored = False
    if st.session_state.awaiting_answer and body_ok and _looks_like_meaningful_answer(user_input):
        fallback = heuristic_result(raw) if tag is None else None
        scored = apply_outcome(tag, fallback, user_input, cleaned_visible)


    new_topic = extract_question_topic(ai_display)
    if new_topic or "?" in ai_display:
        st.session_state.last_question_topic = new_topic
        st.session_state.awaiting_answer = True
    elif scored:
        st.session_state.awaiting_answer = False

    with st.chat_message("assistant"):
        st.markdown(ai_display)

    st.session_state.messages.append(AIMessage(content=ai_display))

#Just a testing block to test the score and the tags in real time in streamlit
"""  

  st.write({
    "awaiting": st.session_state.awaiting_answer,
    "tag": tag,
    "fallback": heuristic_result(raw),
    "body_len": len(cleaned_visible),
    "scored": scored
})

"""