# ─── Part 1 of 4: Imports, Config, JSON I/O, State, Retrieval & External APIs ───────────────

import os
import json
import time
import random
import threading
import re
import requests
import logging
import traceback
from collections import deque, Counter
from datetime import datetime

import numpy as np
import tkinter as tk
from tkinter import simpledialog

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

# Attempt hnswlib
try:
    import hnswlib
except ImportError:
    hnswlib = None

# Attempt Annoy
try:
    from annoy import AnnoyIndex
except ImportError:
    AnnoyIndex = None

# SPARQL & Wikipedia for KB
from SPARQLWrapper import SPARQLWrapper, JSON
import wikipedia

# ─── External KB Helpers ────────────────────────────────────────────────────
def get_factbook_summary(topic):
    try:
        resp = requests.get(
            f"https://restcountries.com/v3.1/name/{topic}",
            timeout=5
        )
        info = resp.json()[0]
        return (
            f"{info['name']['common']} is in {info['region']} "
            f"with population ~{info['population']}."
        )
    except:
        return None

def get_openlibrary_summary(topic):
    try:
        r = requests.get(
            f"https://openlibrary.org/search.json?q={topic}",
            timeout=5
        ).json()
        docs = r.get("docs")
        if docs:
            doc   = docs[0]
            title = doc.get("title", "Unknown")
            auth  = doc.get("author_name", ["Unknown"])[0]
            return f"'{title}' by {auth} appears in OpenLibrary."
    except:
        pass
    return None

def get_wikipedia(q):
    try:
        return wikipedia.summary(q, sentences=2)
    except:
        return None

def get_wikidata(q):
    try:
        sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
        sparql.setQuery(f"""
SELECT ?desc WHERE {{
  ?item rdfs:label "{q}"@en.
  ?item schema:description ?desc.
  FILTER(lang(?desc)="en")
}}
LIMIT 1
""")
        sparql.setReturnFormat(JSON)
        res = sparql.query().convert()["results"]["bindings"]
        return res[0]["desc"]["value"] if res else None
    except:
        return None

def get_kb_answer(q):
    for fn in (
        get_wikipedia,
        get_wikidata,
        get_factbook_summary,
        get_openlibrary_summary
    ):
        try:
            ans = fn(q)
            if ans:
                return ans
        except:
            continue
    return None

# ─── Configuration & Weather API ───────────────────────────────────────────
MEMORY_FILE     = "memory.json"
METRICS_FILE    = "metrics.json"
FEEDBACK_FILE   = "pending_review.json"
AUDIT_FILE      = "audit.json"

Y_OFFSET        = 200
IDLE_THRESHOLD  = 60
ACTIVE_INTERVAL = 3600
AUDIT_INTERVAL  = 86400

# OpenWeatherMap
OWM_API_KEY = "9055430fbd548da87b5c91cbd5f06968"
OWM_URL     = "https://api.openweathermap.org/data/2.5/weather"

def get_weather_data(city):
    """Call OpenWeatherMap and return (data_dict, error_message)."""
    try:
        r = requests.get(
            OWM_URL,
            params={"q": city, "appid": OWM_API_KEY, "units": "metric"},
            timeout=5
        )
        data = r.json()
        if data.get("cod") != 200:
            return None, data.get("message", "API error")
        return data, None
    except Exception as e:
        return None, str(e)

def format_weather(data, unit="Celsius"):
    """Format the raw weather data into a human-readable string."""
    if not data:
        return "No weather data."
    temp_c = data["main"]["temp"]
    desc   = data["weather"][0]["description"].capitalize()
    if unit == "Fahrenheit":
        temp = temp_c * 9/5 + 32
        u    = "°F"
    elif unit == "Kelvin":
        temp = temp_c + 273.15
        u    = "K"
    else:
        temp = temp_c
        u    = "°C"
    return f"{desc}. Temperature: {temp:.1f}{u}."

# ─── JSON I/O Helpers ─────────────────────────────────────────────────────────
def load_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except:
        return default

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

# ─── Persistent State ─────────────────────────────────────────────────────────
memory               = load_json(MEMORY_FILE, {
    "greeting": [], "farewell": [], "thanks": [],
    "default":  [], "weather": [],   "knowledge": []
})
feedback_scores      = {}
conversation_history = []
feedback_context     = {}
weather_calls        = deque()
last_activity        = time.time()

for intent in memory:
    memory.setdefault(intent, [])

# ─── Metrics & Logging ───────────────────────────────────────────────────────
metrics = Counter()
def incr(metric, amt=1):
    metrics[metric] += amt

def flush_metrics():
    save_json(METRICS_FILE, metrics)

threading.Thread(
    target=lambda: (time.sleep(AUDIT_INTERVAL), flush_metrics()),
    daemon=True
).start()

logging.basicConfig(filename="analytics.log", level=logging.INFO)
def log_event(evt, **data):
    logging.info(f"{datetime.now().isoformat()} | {evt} | {data}")

# ─── Active Learning & Audit Loops ──────────────────────────────────────────
def active_learning():
    while True:
        time.sleep(ACTIVE_INTERVAL)
        low = sorted(feedback_scores.items(), key=lambda x: x[1])[:10]
        save_json(FEEDBACK_FILE, dict(low))

threading.Thread(target=active_learning, daemon=True).start()

def audit_export():
    while True:
        time.sleep(AUDIT_INTERVAL)
        save_json(AUDIT_FILE, conversation_history[-100:])

threading.Thread(target=audit_export, daemon=True).start()

# ─── Build Retrieval Indexes ─────────────────────────────────────────────────
corpus    = [txt for bucket in memory.values() for txt in bucket]
tfidf_rag = TfidfVectorizer().fit(corpus) if corpus else None
vecs      = tfidf_rag.transform(corpus).toarray().astype("float32") if corpus else None
D, N      = (vecs.shape[1], len(corpus)) if corpus else (0, 0)

if hnswlib and D > 0:
    hnsw_idx = hnswlib.Index(space='cosine', dim=D)
    hnsw_idx.init_index(max_elements=N, ef_construction=200, M=16)
    hnsw_idx.add_items(vecs, np.arange(N))
    hnsw_idx.set_ef(50)
else:
    hnsw_idx = None

if AnnoyIndex and D > 0:
    annoy_idx = AnnoyIndex(D, 'angular')
    for i, v in enumerate(vecs):
        annoy_idx.add_item(i, v)
    annoy_idx.build(10)
else:
    annoy_idx = None

if D > 0:
    try:
        sk_idx = NearestNeighbors(n_neighbors=5, metric='cosine').fit(vecs)
    except:
        sk_idx = None
else:
    sk_idx = None

# ─── Unified Retrieval Function ─────────────────────────────────────────────
def retrieve_facts(method, query, topk=5):
    qv = tfidf_rag.transform([query]).toarray().astype("float32") if tfidf_rag else None

    if method == 'hnsw' and hnsw_idx:
        ids, _ = hnsw_idx.knn_query(qv, k=topk)
        return [corpus[i] for i in ids[0]]

    if method == 'annoy' and annoy_idx:
        ids = annoy_idx.get_nns_by_vector(qv[0], topk)
        return [corpus[i] for i in ids]

    if method == 'sklearn' and sk_idx:
        _, ids = sk_idx.kneighbors(qv, n_neighbors=topk)
        return [corpus[i] for i in ids[0]]

    sims = cosine_similarity(qv, vecs)[0] if qv is not None else []
    idx  = sims.argsort()[-topk:][::-1]
    return [corpus[i] for i in idx]

default_rag = (
    'hnsw' if hnsw_idx else
    'annoy' if annoy_idx else
    'sklearn' if sk_idx else
    None
)

# ─── NLP & Intent Classification ────────────────────────────────────────────
def tokenize(text):
    return re.findall(r"\b\w+\b", text)

intents = {
    "greeting": ["hello","hi","hey","morning"],
    "farewell": ["bye","goodbye","see you","take care"],
    "thanks":   ["thanks","thank you","much appreciated"],
    "weather":  ["weather","forecast","rain","temperature"],
    "knowledge":["what is","define","explain","tell me about"]
}
intent_list = list(intents.keys())
phrases     = [ph for vs in intents.values() for ph in vs]
labels      = [k for k,v in intents.items() for _ in v]

tfidf_intent = TfidfVectorizer().fit(phrases)
X_intent     = tfidf_intent.transform(phrases)

def classify_intent(txt):
    sims = cosine_similarity(
        tfidf_intent.transform([txt.lower()]), X_intent
    )[0]
    idx = sims.argmax()
    return labels[idx] if sims[idx] >= 0.3 else "default"
# ─── Part 2 of 4: Seeding, Paraphrasing, Evolution, Grammar, KB, Weather & Handlers ─────

import pickle
import random
import time
import threading
import traceback
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neural_network import MLPClassifier
import tkinter as tk
from tkinter import simpledialog

# 🧠 Optional OLD grammar engine
try:
    import language_tool_python
    grammar_tool = language_tool_python.LanguageTool('en-US')
except ImportError:
    grammar_tool = None

# 📚 External KB sources from Part 1:
# get_wikipedia, get_wikidata, get_factbook_summary, get_openlibrary_summary, get_kb_answer

# ─── Seed Responses & Memory Integration ────────────────────────────────────
responses = {
    "greeting": ["Hello!", "Hi there!", "Hey!"],
    "farewell": ["Goodbye!", "Take care!", "See you soon!"],
    "thanks":   ["You're welcome!", "Glad to help!", "Anytime!"],
    "default":  ["Hmm, I didn’t quite get that.", "Can you rephrase?"],
    "weather":  ["Please click 'City' to enter a location for weather."],
    "knowledge":["Ask me about a topic and I'll explain."]
}
for intent in memory:
    responses.setdefault(intent, []).extend(memory[intent])

all_resps = [r for lst in responses.values() for r in lst]

# ─── N-GRAM MODELS ────────────────────────────────────────────────────────
cv2 = CountVectorizer(ngram_range=(2,2), token_pattern=r"\b\w+\b").fit(all_resps)
cv3 = CountVectorizer(ngram_range=(3,3), token_pattern=r"\b\w+\b").fit(all_resps)

def build_counts(cv, texts):
    arr   = cv.transform(texts).toarray().sum(axis=0)
    grams = cv.get_feature_names_out()
    model = {}
    for gram,count in zip(grams, arr):
        parts = gram.split()
        prefix, word = tuple(parts[:-1]), parts[-1]
        model.setdefault(prefix, {})[word] = int(count)
    return model

bi_counts  = build_counts(cv2, all_resps)
tri_counts = build_counts(cv3, all_resps)

# ─── MLP RERANKER & IDLE RETRAIN ──────────────────────────────────────────
train_X, train_y = [], []
for idx, intent in enumerate(intent_list):
    for r in responses[intent]:
        train_X.append(r); train_y.append(idx)

mlp_vect = TfidfVectorizer().fit(train_X)
X_mlp    = mlp_vect.transform(train_X)
mlp      = MLPClassifier(hidden_layer_sizes=(128,64,32), max_iter=300, random_state=42)
mlp.fit(X_mlp, train_y)

def train_mlp():
    ex, y = [], []
    for idx, intent in enumerate(intent_list):
        for r in responses[intent]:
            ex.append(r); y.append(idx)
    global mlp_vect, mlp
    mlp_vect = TfidfVectorizer().fit(ex)
    mlp      = MLPClassifier(hidden_layer_sizes=(128,64,32), max_iter=300, random_state=42)
    mlp.fit(mlp_vect.transform(ex), y)

def idle_retrain():
    global last_activity
    while True:
        time.sleep(5)
        if time.time() - last_activity >= IDLE_THRESHOLD:
            train_mlp()

threading.Thread(target=idle_retrain, daemon=True).start()

# ─── TEXT QUALITY CHECKS ──────────────────────────────────────────────────
def grammar_check(text):
    basic = len(tokenize(text)) >= 5
    if grammar_tool:
        matches = grammar_tool.check(text)
        return basic and len(matches) < 2
    return basic

def coherence_check(text):
    ws = tokenize(text)
    return len(ws)>1 and len(set(ws))/len(ws) >= 0.5

def comprehension_check(user, reply):
    return len(set(tokenize(user.lower())) & set(tokenize(reply.lower()))) >= 2

# ─── PARAPHRASING ENGINE ──────────────────────────────────────────────────
local_synonyms = {
    "good":    ["great","excellent","fine"],
    "help":    ["assist","aid","support"],
    "weather": ["forecast","climate","conditions"],
    "see":     ["view","notice","observe"]
}
flair_templates = [
    lambda t: f"Let me rephrase that: {t}",
    lambda t: f"Another way to say it — {t}",
    lambda t: f"Think of it this way: {t}"
]

def paraphrase(txt):
    ws  = tokenize(txt)
    out = []
    for w in ws:
        opts = local_synonyms.get(w.lower(), [])
        out.append(random.choice(opts) if opts and random.random()<0.3 else w)
    phrased = " ".join(out)
    return random.choice(flair_templates)(phrased)

# ─── EVOLUTIONARY GENERATION ─────────────────────────────────────────────
def generate_raw_response(intent, max_len=50):
    base = responses.get(intent) or responses["default"]
    seed = random.choice(base).split()[:2]
    wds  = seed.copy()
    for _ in range(max_len - len(wds)):
        tri_k = tuple(wds[-2:])
        if tri_k in tri_counts:
            ch, wt = zip(*tri_counts[tri_k].items())
        else:
            bi_k = (wds[-1],)
            if bi_k not in bi_counts: break
            ch, wt = zip(*bi_counts[bi_k].items())
        wds.append(random.choices(ch, weights=[w/sum(wt) for w in wt])[0])
    return " ".join(wds)

def mutate_response(resp):
    ws = resp.split()
    if not ws: return resp
    i  = random.randrange(len(ws))
    pfx= tuple(ws[max(0,i-2):i])
    model = tri_counts.get(pfx) or bi_counts.get((ws[i-1],), {})
    if not model: return resp
    ws[i] = random.choices(list(model), weights=list(model.values()))[0]
    return " ".join(ws)

def fitness(r, intent, ui):
    g    = grammar_check(r)
    c    = coherence_check(r)
    p    = comprehension_check(ui, r)
    nov  = 1 - max(cosine_similarity(mlp_vect.transform([r]), mlp_vect.transform(all_resps))[0])
    prob = mlp.predict_proba(mlp_vect.transform([r]))[0][intent_list.index(intent)]
    rw   = feedback_scores.get(r, 0)/5.0
    return (g + c + p + nov + prob + rw) / 6

def evolve_response(intent, ui, pop=10, gens=3):
    popu = [generate_raw_response(intent) for _ in range(pop)]
    for _ in range(gens):
        scored  = sorted(((fitness(r,intent,ui),r) for r in popu), reverse=True)
        parents = [r for _,r in scored[:pop//2]]
        popu    = parents[:]
        while len(popu)<pop:
            popu.append(mutate_response(random.choice(parents)))
    return max(popu, key=lambda r: fitness(r,intent,ui))

def generate_valid_response(intent, ui):
    if intent in ("greeting","farewell","thanks"):
        return random.choice(responses[intent])
    if intent == "knowledge":
        ans = get_kb_answer(ui.lower())
        return ans or paraphrase(random.choice(responses["default"]))
    if intent == "weather":
        return "Click 'City' to enter a location."
    return evolve_response(intent, ui)

def explain(r, intent, ui):
    pts = []
    if grammar_check(r):          pts.append("✓ Grammar")
    if coherence_check(r):        pts.append("✓ Coherence")
    if comprehension_check(ui,r):  pts.append("✓ Relevant")
    if feedback_scores.get(r,0)>0: pts.append("✓ Approved")
    return " • ".join(pts)

# ─── WEATHER INTEGRATION ───────────────────────────────────────────────────
# uses get_weather_data() & format_weather() from Part 1

last_weather_data = None
last_city         = None

def fetch_and_store(city):
    data, err = get_weather_data(city)
    if err:
        return None, f"Error fetching weather: {err}"
    global last_weather_data, last_city
    last_weather_data = data
    last_city         = city
    return data, f"Loaded weather for {city}. Click 'Convert' to view."

def on_city():
    city = simpledialog.askstring("City", "Enter city:")
    if not city:
        return
    _, msg = fetch_and_store(city)
    chat.insert(tk.END, f"Bot: {msg}\n\n")

def on_convert():
    if last_weather_data is None:
        chat.insert(tk.END, "Bot: No weather data loaded. Click 'City' first.\n\n")
        return
    res = format_weather(last_weather_data, unit_var.get())
    chat.insert(tk.END, f"Bot: {res}\n\n")

# ─── FEEDBACK HANDLER ─────────────────────────────────────────────────────
def log_feedback(delta):
    r = feedback_context.get("response")
    if r:
        feedback_scores[r] = feedback_scores.get(r, 0) + delta
        log_event("feedback", response=r, delta=delta)
    chat.insert(tk.END, "Bot: Feedback noted.\n\n")
    feedback_frame.pack_forget()

# ─── TKINTER EVENT HANDLERS ─────────────────────────────────────────────
def on_send():
    global last_activity
    text = entry.get().strip()
    if not text:
        return
    entry.delete(0, tk.END)
    chat.insert(tk.END, f"You: {text}\n")
    last_activity = time.time()

    intent = classify_intent(text)
    resp   = generate_valid_response(intent, text)

    chat.insert(tk.END, f"Bot: {resp}\n{explain(resp,intent,text)}\n\n")
    feedback_context["response"] = resp
    feedback_frame.pack(pady=5)
    conversation_history.append({"user": text, "bot": resp})
    # ─── Continued Part 2: Remaining Event Handlers & Teach Window ───────────────────────────

def on_weather_cancel():
    """Handler to cancel a pending weather request."""
    chat.insert(tk.END, "Bot: Weather canceled.\n\n")

def toggle_weather_ui():
    """Show or hide the city/unit frames for weather input."""
    global weather_ui_visible
    if weather_ui_visible:
        city_frame.pack_forget()
        unit_frame.pack_forget()
    else:
        city_frame.pack(pady=5)
        unit_frame.pack(pady=5)
    weather_ui_visible = not weather_ui_visible

def open_teach_window():
    """Pop up a window so the user can teach the bot a new concept."""
    teach = tk.Toplevel(root)
    teach.title("Teach Me Something")
    tk.Label(teach, text="Enter a new concept or phrase:").pack(pady=5)
    teach_input = tk.Entry(teach, width=40)
    teach_input.pack(pady=5)
    def save_concept():
        topic = teach_input.get().strip()
        if topic:
            memory.setdefault("knowledge", []).append(topic)
            save_json(MEMORY_FILE, memory)
            chat.insert(tk.END, f"Bot: Got it! Added '{topic}' to my knowledge.\n\n")
            teach.destroy()
    tk.Button(teach, text="Teach", command=save_concept).pack(pady=10)
  # ─── Part 3 of 4: GUI Setup with Proper Weather Toggle & safe_call ─────────────

import tkinter as tk
import time
import traceback

# assume all handlers (on_send, on_city, on_convert, on_weather_cancel,
# open_teach_window, log_feedback) have been defined in Part 2

# ─── safe_call decorator ─────────────────────────────────────────────────────
def safe_call(fn):
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            chat.insert(tk.END, f"⚠️ Error in {fn.__name__}: {e}\n")
            chat.insert(tk.END, traceback.format_exc() + "\n")
    return wrapper

root = tk.Tk()
root.title("SmartWeatherBot")

# ─── Chat Display ─────────────────────────────────────────────────────────
chat = tk.Text(root, height=18, width=60, bg="black", fg="white")
chat.pack(padx=10, pady=(10,5))

# ─── Context Menu for Chat ────────────────────────────────────────────────
ctx = tk.Menu(root, tearoff=0)
ctx.add_command(label="Copy Selected", command=lambda: (
    root.clipboard_clear(),
    root.clipboard_append(chat.selection_get() if chat.tag_ranges("sel") else "")
))
ctx.add_command(label="Copy All Errors", command=lambda: (
    root.clipboard_clear(),
    root.clipboard_append("\n".join(
        [l for l in chat.get("1.0", tk.END).splitlines() if l.startswith("⚠️")]
    ))
))
chat.bind("<Button-3>", lambda e: ctx.tk_popup(e.x_root, e.y_root))

# ─── User Entry & Send Button ─────────────────────────────────────────────
entry = tk.Entry(root, width=50)
entry.pack(pady=5)

# wrap on_send after defining safe_call
on_send = safe_call(on_send)
tk.Button(root, text="Send", command=on_send).pack()

# ─── Feedback Frame (hidden until after a response) ────────────────────────
feedback_frame = tk.Frame(root)
tk.Button(feedback_frame, text="👍 Keep This",   command=lambda: safe_call(log_feedback)(+1)).pack(side=tk.LEFT)
tk.Button(feedback_frame, text="📝 Improve This",command=lambda: safe_call(log_feedback)(-1)).pack(side=tk.LEFT)

# ─── Reworked Weather UI ──────────────────────────────────────────────────
weather_ui_visible = False
last_weather_data  = None
last_city          = None

weather_frame = tk.Frame(root)

# City input sub-frame
city_frame = tk.Frame(weather_frame)
tk.Label(city_frame, text="City:", font=("Arial",10)).pack(side=tk.LEFT)
city_entry = tk.Entry(city_frame, width=20)
city_entry.pack(side=tk.LEFT, padx=5)

# Define actual on_city_ui to use city_entry.get()
def on_city_ui():
    city = city_entry.get().strip()
    if not city:
        chat.insert(tk.END, "Bot: Please enter a city name.\n\n")
        return
    data, msg = fetch_and_store(city)
    chat.insert(tk.END, f"Bot: {msg}\n\n")
    if data:
        unit_frame.pack(pady=5)

# wire and wrap on_city_ui
on_city_ui = safe_call(on_city_ui)
tk.Button(city_frame, text="OK",     command=on_city_ui).pack(side=tk.LEFT)
tk.Button(city_frame, text="Cancel", command=safe_call(on_weather_cancel)).pack(side=tk.LEFT)

# Unit selection & convert sub-frame
unit_frame = tk.Frame(weather_frame)
tk.Label(unit_frame, text="Unit:", font=("Arial",10)).pack(side=tk.LEFT)
unit_var = tk.StringVar(value="Celsius")
for u in ("Celsius","Fahrenheit","Kelvin"):
    tk.Radiobutton(unit_frame, text=u, variable=unit_var, value=u).pack(side=tk.LEFT, padx=5)
tk.Button(unit_frame, text="Convert", command=safe_call(on_convert)).pack(side=tk.LEFT)

# Define and wrap toggle_weather_ui before creating its button
def toggle_weather_ui():
    global weather_ui_visible
    if weather_ui_visible:
        weather_frame.pack_forget()
    else:
        weather_frame.pack(pady=5)
        city_frame.pack(pady=5)
    weather_ui_visible = not weather_ui_visible

toggle_weather_ui = safe_call(toggle_weather_ui)
tk.Button(root, text="Weather", command=toggle_weather_ui).pack(pady=5)

# ─── Optional “Teach” Button ──────────────────────────────────────────────
open_teach_window = safe_call(open_teach_window)
tk.Button(root, text="Teach", command=open_teach_window).pack(pady=(0,10))

# ─── Centering & Typing Animation Helpers ─────────────────────────────────
def center(win):
    win.update_idletasks()
    x = (win.winfo_screenwidth() - win.winfo_width()) // 2
    y = (win.winfo_screenheight() - win.winfo_height()) // 2 + Y_OFFSET
    win.geometry(f"{win.winfo_width()}x{win.winfo_height()}+{x}+{y}")

entry.bind("<FocusIn>", lambda e: center(root))
center(root)

# ─── Launch GUI ─────────────────────────────────────────────────────────
try:
    root.mainloop()
except Exception as e:
    print("App crashed:", e)
    # ─── Part 4 of 4: Persistence & Graceful Shutdown ─────────────────────

# Auto‐save memory and metrics every 5 minutes
def auto_save():
    while True:
        time.sleep(300)
        save_json(MEMORY_FILE, memory)
        flush_metrics()

threading.Thread(target=auto_save, daemon=True).start()

# Graceful exit: save all state on window close
def on_shutdown():
    save_json(MEMORY_FILE, memory)
    save_json(METRICS_FILE, metrics)
    save_json(FEEDBACK_FILE, feedback_scores)
    save_json(AUDIT_FILE, conversation_history)
    root.destroy()

# Register the shutdown handler before mainloop
root.protocol("WM_DELETE_WINDOW", on_shutdown)
import resource

def print_mem_usage():
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"📊 Memory usage: {usage / 1024:.2f} MB")