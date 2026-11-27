"""Prompts of the question-generation routes (generate_questions.py)."""

# ---------------------------------------------------------------- shared rules
SELF_CONTAINED = (
    "Every question must be fully self-contained: name the entities explicitly (never 'he', 'she', 'it', "
    "'they', and never 'the document' or 'the passage'), so that it makes sense without seeing the "
    "documents. Each question must have a SHORT factual answer (a few words) stated in the documents. "
    "Exactly one question per line, never two questions joined by 'and'. "
)
DOC_PREFIX = (
    "After thinking, output ONLY the questions, one per line, each prefixed with the document number(s) it "
    "needs, like 'D7: ...?' (or 'D3+D11: ...?' for multi-document questions). No numbering, no other text."
)
STORY_RULES = (
    "Every question must be fully self-contained and understandable without the text: always name the "
    "characters (never 'he', 'she', 'they', 'the narrator', 'the man'; describe an unnamed character by "
    "role, e.g. 'the innkeeper'); never refer to 'this part', 'the passage', 'the excerpt' or 'the text' "
    "(you may say 'in {title}'). Each answer must be SHORT (a name, a place, an object, a number or a "
    "phrase of a few words) and stated in the text. Exactly one question per line, never two questions "
    "joined by 'and'. Skip publication metadata, tables of contents, cast lists and licensing text. "
)
Q_PREFIX = (
    "After thinking, output ONLY the questions, one per line, each prefixed 'Q: '. No numbering, no "
    "answers, no other text."
)
DOCUMENT_RULES = (
    "Every question must be self-contained: name the subject explicitly (never 'he', 'it', 'this'); since "
    "there is a single document you may say 'the document', 'the paper', 'the report' or 'the article', "
    "but never 'this part', 'the passage' or 'the excerpt'. Each answer must be short (a name, number, "
    "date, term or a phrase of a few words) and stated in the text. Exactly one question per line, never "
    "two questions joined by 'and'. "
)
PAGE_RULES = (
    "Every question must be fully self-contained: name the page's subject and the other entities "
    "explicitly (never 'he', 'she', 'it', 'they', and never 'the page', 'the article', 'this part' or 'the "
    "text'), so that it makes sense without seeing the page. Each question must have a SHORT factual "
    "answer (a name, date, number, place or a phrase of a few words) stated in the page. Each line must be "
    "a complete question ending with a question mark (never a bare noun phrase such as 'Host country of "
    "X'). Exactly one question per line, never two questions joined by 'and'. "
)

# ---------------------------------------------------------------- one prompt per route, by recipe
PROMPTS = {
    "musique": {
        "S": (
            "Below are {n} documents.\n\n{docs}\n\nFor EACH document, in order from Document 1 to Document "
            "{n}, write exactly 2 questions that can be answered from that document alone. Cover different "
            "facts (people, places, dates, numbers, works, relations). "
            + SELF_CONTAINED +
            "While thinking, go through the documents one by one. "
            + DOC_PREFIX
        ),
        "B": (
            "Below are {n} documents.\n\n{docs}\n\nWrite 10 MULTI-HOP questions, each requiring facts from "
            "two or three different documents, in the style of the MuSiQue dataset. The bridge entity must "
            "NOT be named: refer to it only through a description that another document supports. Good "
            "examples: 'Who is the spouse of the performer of Green?' (document A says Green was performed "
            "by X, document B says X's spouse is Y -> answer Y); 'In which country is the birthplace of "
            "the author of Ulysses?'; 'What is the population of the city where the company that makes the "
            "iPhone is headquartered?'. BAD (never do this): questions about the documents themselves such "
            "as 'Which document states that ...?' or 'What is mentioned in ...?'; the word 'document' must "
            "not appear in any question. The final answer must be a short phrase stated in one of the "
            "documents. While thinking, first list entities that appear in more than one document, then "
            "build the chains. "
            + SELF_CONTAINED
            + DOC_PREFIX
        ),
    },
    "2wiki": {
        "S": (
            "Below are {n} documents.\n\n{docs}\n\nFor EACH document, in order from Document 1 to Document "
            "{n}, write exactly 3 questions that can be answered from that document alone (birth/death "
            "dates and places, nationality, occupation, works, family relations, directors, "
            "publishers...). "
            + SELF_CONTAINED +
            "While thinking, go through the documents one by one. "
            + DOC_PREFIX
        ),
        "R": (
            "Below are {n} documents.\n\n{docs}\n\nWrite 10 {style} Each question must combine at least "
            "two documents. "
            + SELF_CONTAINED
            + DOC_PREFIX
        ),
    },
    "qasper": {
        "S": (
            "Below is a scientific paper split into sections.\n\n"
            "{docs}\n\nFor each substantive section (skip sections that only list references or contain no "
            "concrete facts), write 2 or 3 questions a reader might ask about the paper that this section "
            "answers with a SHORT answer (a few words): datasets and their sizes, metrics, baselines, "
            "model components, hyperparameters, languages, experimental settings, results. Since the paper "
            "is fixed, you may say 'the paper', 'the authors' or 'they'. At most 40 questions in total, "
            "one per line. After thinking, output ONLY the questions, each prefixed with its section "
            "number like 'S4: ...?'. No numbering, no other text."
        ),
        "E": (
            "Below is a scientific paper split into sections.\n\n"
            "{docs}\n\nWrite 10 questions about the paper whose short answer (a phrase, a number, a list "
            "of names) must be located by reading across sections: which evaluation metrics are used, on "
            "which dataset the improvement is largest, which baselines are compared, which languages are "
            "covered, what the training data size is, and so on. Make each question specific to THIS paper "
            "by naming the datasets, models, tasks or languages involved (do not reuse the generic wording "
            "above). You may say 'the paper' or 'they'. After thinking, output ONLY the questions, one per "
            "line, prefixed 'E: '. No numbering, no other text."
        ),
        "Y": (
            "Below is a scientific paper split into sections.\n\n"
            "{docs}\n\nWrite 6 YES/NO questions about the paper's method, data and evaluation (e.g. 'Do "
            "they use pretrained word embeddings?', 'Is a human evaluation performed?'). Exactly 3 must "
            "have the answer yes and 3 the answer no according to the paper. After thinking, output ONLY "
            "the questions, one per line, prefixed 'Y: '. No numbering, no answers, no other text."
        ),
        "A": (
            "Below is a scientific paper split into sections.\n\n"
            "{docs}\n\nWrite 4 questions whose answer is a short abstractive statement (one clause, not a "
            "verbatim quote): the main contribution, the key difference from prior work or baselines, the "
            "main limitation or failure case the authors mention, the main finding. After thinking, output "
            "ONLY the questions, one per line, prefixed 'A: '. No numbering, no other text."
        ),
    },
    "narrativeqa": {
        "L1": (
            "Below is part {i} of {n} of the story \"{title}\".{prev}\n\n"
            "=== Part {i} ===\n{chunk}\n\nWrite exactly 15 questions about the EVENTS in part {i}: who "
            "does what, to whom, where, with what, what happens next, how something turns out. Cover the "
            "whole part from beginning to end, in order. "
            + STORY_RULES
            + Q_PREFIX
        ),
        "L2": (
            "Below is part {i} of {n} of the story \"{title}\".{prev}\n\n"
            "=== Part {i} ===\n{chunk}\n\nWrite exactly 15 questions about the CHARACTERS AND DETAILS in "
            "part {i}: who a character is (role, occupation, relationship to other characters), what a "
            "character says, wants, thinks or feels, why a character acts as they do (short-answer "
            "why/how), and descriptions of places, objects and appearance. "
            + STORY_RULES
            + Q_PREFIX
        ),
        "G": (
            "Below is the complete story \"{title}\" in {n} parts.\n\n"
            "{docs}\n\nWrite 40 questions about the story as a whole, like reading-comprehension questions "
            "about the plot: 14 about who the characters are and how they are related (someone's father, "
            "occupation, who marries whom, who kills whom); 10 about motives and manner (why a character "
            "does something, how a character achieves something); 6 about outcomes and the ending (what "
            "happens to a character in the end, how someone dies); 4 about the setting (where and when the "
            "story takes place, where a character lives); 6 about causality and order across the story "
            "(what leads a character to do something, what happens after a given event). "
            + STORY_RULES +
            "After thinking, output ONLY the questions, one per line, each prefixed 'G: '. No numbering, "
            "no answers, no other text."
        ),
    },
    "multifieldqa": {
        "S": (
            "Below is part {i} of {n} of a document. For context only, the document begins:\n\"{head}\"\n\n"
            "=== Part {i} ===\n{chunk}\n\nWrite exactly 5 questions that can be answered from part {i} "
            "alone: names, numbers, dates, organisations, methods, definitions, provisions, findings. "
            + DOCUMENT_RULES
            + Q_PREFIX
        ),
        "E": (
            "Below is a document split into parts.\n\n{docs}\n\n"
            "Write 10 questions about the document as a whole whose short answer must be located by "
            "reading across parts: the topic and purpose of the document, its main conclusions or "
            "outcomes, the methods, techniques or procedures used, key numbers, and the people or "
            "organisations involved. Make each question specific to THIS document by naming its subject. "
            + DOCUMENT_RULES +
            "After thinking, output ONLY the questions, one per line, each prefixed 'E: '. No numbering, "
            "no answers, no other text."
        ),
        "A": (
            "Below is a document split into parts.\n\n{docs}\n\n"
            "Write 5 explanatory questions (why / how / what is the effect of / what is the purpose of) "
            "whose answer is ONE short clause stated in the document, e.g. 'Why is the fuselage sloped?', "
            "'How does X affect Y?'. "
            + DOCUMENT_RULES +
            "After thinking, output ONLY the questions, one per line, each prefixed 'A: '. No numbering, "
            "no answers, no other text."
        ),
    },
    "squad": {
        "S": (
            "Below is a paragraph from the Wikipedia article \"{title}\".\n\n"
            "{chunk}\n\nWrite exactly 20 questions that can be answered from this paragraph alone. First "
            "cover every distinct fact in the paragraph (people, organisations, dates, numbers, places, "
            "events, causes, definitions); once the facts are exhausted, ask about the same facts from "
            "different angles (who / what / when / where / how many / why / which). "
            + SELF_CONTAINED +
            "Never refer to 'the paragraph', 'the passage', 'the text' or 'the article'. "
            + Q_PREFIX
        ),
    },
    "nq": {
        "G": (
            "Below is the Wikipedia page \"{title}\" split into {n} parts.\n\n"
            "{docs}\n\nWrite 20 questions about \"{title}\" that a search-engine user would most likely "
            "ask: what or who the subject is; key dates (founded, born, released, held, ended); key people "
            "(founder, director, author, winner, holder of a role); places; key numbers (population, "
            "episodes, length, box office, score); outcomes and superlatives (who won, the first, the "
            "last, the most). Every answer must be stated in the page. "
            + PAGE_RULES +
            "After thinking, output ONLY the questions, one per line, each prefixed 'G: '. No numbering, "
            "no answers, no other text."
        ),
        "Q": (
            "Below is the Wikipedia page \"{title}\" split into {n} parts.\n\n"
            "{docs}\n\nWrite 15 search queries that a user might type into a search engine and that this "
            "page answers, in the style of real search queries: all lowercase, no question mark, natural "
            "wording, e.g. 'when was the first transatlantic telegraph cable laid', 'who invented the "
            "safety pin', 'how tall is mount everest in feet'. Cover the facts a user would most likely "
            "look up about \"{title}\": what or who the subject is, key dates, key people, places, key "
            "numbers, outcomes and superlatives. Each query must name the subject explicitly so it makes "
            "sense on its own (never 'he', 'she', 'it', 'they', 'the page' or 'the article') and must have "
            "a SHORT factual answer (a few words) stated in the page. Exactly one query per line. After "
            "thinking, output ONLY the queries, one per line, each prefixed 'Q: '. No numbering, no "
            "answers, no other text."
        ),
        "L": (
            "Below is part {i} of {n} of the Wikipedia page \"{title}\". For context only, the page "
            "begins:\n\"{head}\"\n\n=== Part {i} ===\n{chunk}\n\n"
            "Write exactly 12 questions that can be answered from part {i} alone, covering its facts in "
            "order: people, dates, numbers, places, positions and roles, works, relationships, and entries "
            "in its lists or tables. Skip reference lists, citations ('Retrieved ...'), categories, "
            "navigation links and other boilerplate; if part {i} is mostly boilerplate, write fewer "
            "questions. "
            + PAGE_RULES +
            "After thinking, output ONLY the questions, one per line, each prefixed 'L: '. No numbering, "
            "no answers, no other text."
        ),
    },
}
PROMPTS["hotpotqa"] = {"S": PROMPTS["2wiki"]["S"], "R": PROMPTS["2wiki"]["R"]}

# previous part of a story, shown as context in the per-part story prompts
PREVIOUS_PART = (
    "\n\nFor context only (do NOT write questions about it), here is the previous part:\n=== Part {i} "
    "(context) ===\n{text}"
)

# {style} of the typed multi-document route (R), by the question type of the unit
STYLES = {
    "2wiki": {
        "comparison": (
            "COMPARISON questions: compare two entities from two different documents on the same attribute "
            "(which was born earlier / died later, which is longer, are they from the same country...). "
            "Make 5 of them yes/no questions ('Are X and Y both from Germany?', answer exactly yes or no) "
            "and 5 'which one' questions whose answer is one of the two entity names."
        ),
        "bridge_comparison": (
            "BRIDGE-COMPARISON questions: compare two entities that are reached through a bridge from "
            "another document, and refer to them ONLY by the bridge, never by name. Example: 'Which film "
            "has the director who was born earlier, Playthings of Passion or The Penalty?' (the directors' "
            "names must not appear; the answer is one of the two film titles). Another: 'Which film's "
            "director died later, X or Y?'. Vary the attribute (born earlier/later, died earlier/later, "
            "nationality)."
        ),
        "compositional": (
            "COMPOSITIONAL two-hop questions: an attribute of an entity that is itself identified through "
            "another document, and that entity must NOT be named — only described (e.g. 'Where was the "
            "spouse of X born?', 'Who is the father of the director of film Y?', 'What is the date of "
            "death of the composer of Z?'). The answer is a short phrase from the second document."
        ),
        "inference": (
            "INFERENCE questions about family or relation chains that must be combined across documents "
            "(e.g. 'Who is the paternal grandfather of X?', 'Who is the sibling-in-law of Y?'). The answer "
            "is a person's name."
        ),
    },
    "hotpotqa": {
        "bridge": (
            "BRIDGE two-hop questions in the style of HotpotQA: ask about an attribute of an entity that "
            "is identified only through another document, and do NOT name that bridge entity. Example: "
            "'What government position was held by the woman who portrayed Corliss Archer in the film Kiss "
            "and Tell?' (document A: the actress in that film was X; document B: X held position Y -> "
            "answer Y). Other patterns: 'In which year was the founder of the company that makes Z born?', "
            "'Which league does the team that plays at stadium W belong to?'. The answer is a short phrase "
            "from the second document."
        ),
    },
}
STYLES["hotpotqa"]["comparison"] = STYLES["2wiki"]["comparison"]

# conversation routes (LoCoMo)
LOCOMO = {
    "single_hop": (
        "Below is one session from a long-term conversation between {A} and {B}.\nThis session took place "
        "on {date}.\n\n[SESSION]\n{session}\n\nGenerate {N} diverse questions that test memory of the "
        "facts in this session.\nRequirements:\n- Each question must be answerable solely from this "
        "session, with a short factual answer (a few words).\n- Each question must be fully "
        "self-contained: always use the speakers' names (never \"he\", \"she\" or \"they\"), and anchor "
        "the question with the date or a concrete event whenever it could otherwise be ambiguous, e.g. "
        "\"What gift did {A} buy for her daughter around {date}?\".\n- Cover as many DIFFERENT facts as "
        "possible: events, people, places, objects, numbers, dates, feelings, opinions, plans, and any "
        "photos they shared.\n- Vary the style: what / who / when / where / why / how, plus a few yes/no "
        "questions.\nWhile thinking, first enumerate the distinct facts of the session, then compose the "
        "questions. After thinking, output ONLY the questions, one per line, no numbering."
    ),
    "multi_hop": (
        "Below is a long-term conversation between {A} and {B}, spanning multiple sessions on different "
        "dates. The date of each session is written before it.\n\n"
        "[FULL CONVERSATION]\n{conv}\n\nGenerate {N} MULTI-HOP questions that can ONLY be answered by "
        "COMBINING information from at least two different sessions. While thinking, first collect facts "
        "that recur or connect across sessions, then compose the questions. Mix these styles:\n- "
        "aggregation: \"How many times did {A} mention ...?\", \"List all the ... that {B} talked about "
        "across the conversation.\"\n- cross-reference: the same person/thing appearing in different "
        "sessions, e.g. \"What did {A} say about her mentor when the topic came up again later?\"\n- "
        "change: \"How did {B}'s opinion about X change over the conversation?\"\nDo NOT ask about dates, "
        "time order or durations — those belong to another category.\nHard requirements: fully "
        "self-contained (explicit speaker names, concrete event anchors, never \"he/she/they\"); each has "
        "a short factual answer stated in the conversation. Each line must be EXACTLY ONE question — never "
        "two questions joined by \"and\", and no narrative preamble before the question mark; the "
        "cross-session combination must live inside the single question itself (as in the aggregation and "
        "cross-reference styles above), not in a second clause. After thinking, output ONLY the questions, "
        "one per line, no numbering."
    ),
    "temporal": (
        "Below is a long-term conversation between {A} and {B}, spanning multiple sessions on different "
        "dates. The date of each session is written before it.\n\n"
        "[FULL CONVERSATION]\n{conv}\n\nGenerate {N} TEMPORAL questions whose answers are dates, orderings "
        "or durations, derivable from the session dates. Mix these styles:\n- \"On what date did X "
        "happen?\"\n- \"What did {A} do before/after Y?\"\n- \"How long after X did Y happen?\"\n- \"In "
        "which month did {B} ...?\"\nHard requirements: fully self-contained (explicit speaker names, "
        "concrete event anchors); each answer is a short date, a duration, or an event name. After "
        "thinking, output ONLY the questions, one per line, no numbering."
    ),
    "open_domain": (
        "Below is a long-term conversation between {A} and {B}, spanning multiple sessions on different "
        "dates.\n\n[FULL CONVERSATION]\n{conv}\n\nGenerate {N} OPEN-DOMAIN questions: each one requires "
        "combining clues from the conversation with common sense or world knowledge, and its answer may "
        "NOT be stated word-for-word in the conversation. Model them on these patterns:\n- role/identity "
        "induction: \"What kind of job is {A} effectively doing, given her activities?\"\n- plausible "
        "attribute inference: \"What might {B}'s financial situation be?\"\n- world-knowledge bridging: "
        "\"Which real-world park/band/city/book could {A} be referring to?\"\n- preference/personality "
        "inference: \"Would {B} likely enjoy ...? \" (yes/no with an implied reason)\nEach question must "
        "still have a SHORT, defensible answer (a few words) strongly supported by conversation clues, and "
        "must EMBED the supporting clue in the question itself (e.g. \"..., given her family bonding "
        "activities?\"). Never ask ungrounded guessing questions like \"Which real-world book could {A} "
        "have read?\" — if the conversation offers no concrete clue, do not ask. Fully self-contained "
        "wording with explicit speaker names. After thinking, output ONLY the questions, one per line, no "
        "numbering."
    ),
}
