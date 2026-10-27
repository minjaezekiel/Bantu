// ============================================================================
// lugha/corpus/slots.b — the vocabulary the templates draw from
//
// Separated from generate.b so the lists can be read and audited on their own.
// Every list is lowercase: the corpus is stored post-normalisation, so the
// model trains on exactly the surface form inference will hand it.
// ============================================================================

$SLOTS = {
    // ── people and address ──────────────────────────────────────────────
    "you":        ["you", "u", "ya"],
    "name":       ["sam", "alex", "jordan", "riley", "casey", "jamie", "taylor"],
    "relation":   ["mum", "mom", "dad", "parents", "teacher", "sister", "brother",
                   "friends", "family", "aunt"],

    // ── harassment ──────────────────────────────────────────────────────
    "insult":     ["idiot", "moron", "loser", "freak", "clown", "waste of space",
                   "pig", "fool", "creep", "weirdo", "nobody", "failure",
                   "dickhead", "asshole", "arsehole", "wanker", "prick", "twat",
                   "fucker", "motherfucker", "scumbag", "pervert"],
    "insult_adj": ["pathetic", "worthless", "disgusting", "useless", "stupid",
                   "ugly", "revolting", "repulsive", "brainless", "hopeless"],
    "harass_verb":["shut up", "get lost", "crawl back in your hole",
                   "stop existing", "delete your account", "go away"],

    // ── derogatory terms, general-purpose ───────────────────────────────
    // Deliberately limited to widely-recognised profane insults. Ethnic,
    // religious and sexual-orientation slurs are NOT listed here; they live in
    // the operator-maintained lexicon the deterministic stage reads. See
    // corpus/README.md for why publishing such a list would itself be a harm.
    "slur":       ["bitch", "whore", "slut", "retard", "bastard", "cunt",
                   "fag", "tranny", "spastic", "fucking retard", "stupid bitch",
                   "faggot", "mong", "sped", "window licker", "little slag",
                   "dumb whore", "absolute spastic", "braindead retard"],

    // ── sexual, adult ───────────────────────────────────────────────────
    "sex_noun":   ["sex", "nudes", "porn", "dick pics", "my body", "naked photos"],
    "sex_ask":    ["send nudes", "send me a naked photo", "wanna sext",
                   "show me your body", "lets have sex", "wanna hook up tonight",
                   "send me something spicy", "send me a pic with nothing on",
                   "take your top off for me", "show me more",
                   "lets do something filthy", "send me the photos you dont post",
                   "wanna come over for the night", "describe what youd do to me",
                   "send me one without the clothes"],
    "sex_adj":    ["horny", "naked", "topless", "aroused"],

    // ── minor markers ───────────────────────────────────────────────────
    "minor_age":  ["12", "13", "14", "15", "16", "17"],
    "minor_word": ["kid", "kids", "child", "children", "little girl",
                   "little boy", "schoolgirl", "schoolboy", "teen", "minor",
                   "young girl", "young boy", "someone in school",
                   "year nine girl", "year eight boy", "little one",
                   "someone underage", "high schooler"],
    "grade":      ["year 8", "year 9", "grade 7", "grade 8", "middle school",
                   "primary school"],

    // ── grooming ────────────────────────────────────────────────────────
    "groom_secret":["dont tell anyone", "this is our secret", "keep this between us",
                   "dont tell your parents", "delete these messages after",
                   "nobody else needs to know", "clear your history after this",
                   "if anyone asks we never spoke", "this stays between you and me",
                   "dont screenshot this", "promise me you wont tell",
                   "your parents wouldnt understand", "keep me off your phone"],
    "groom_move": ["lets talk on another app", "add me on snap", "whats your number",
                   "lets move this to private chat", "do you have telegram",
                   "message me somewhere else", "do you use whatsapp",
                   "lets video call instead", "add me on the other thing",
                   "give me your number and ill text you",
                   "find me where your mum cant see", "switch to dms"],
    "groom_probe":["are you home alone", "are your parents out", "whats your address",
                   "what school do you go to", "when are you alone",
                   "do your parents check your phone", "whats your room like",
                   "what time do you get home", "who else lives with you",
                   "does anyone read your messages", "are you alone right now",
                   "do you have your own phone", "when is the house empty"],
    "groom_flat": ["youre mature for your age", "youre not like other kids your age",
                   "i feel like you understand me", "youre too pretty to be that young",
                   "youre special, not like the others",
                   "youre wise beyond your years", "nobody gets me like you do",
                   "youre more grown up than people your age",
                   "i can talk to you properly, unlike them",
                   "youd pass for a lot older", "we have a connection",
                   "youre the only one who listens to me"],

    // ── threat ──────────────────────────────────────────────────────────
    "threat_act": ["kill you", "hurt you", "find you", "break your face",
                   "make you pay", "end you", "beat you up", "smash your face in",
                   "put you in hospital", "wreck you", "come for you",
                   "deal with you", "sort you out", "snap your neck",
                   "burn your house down", "make you regret it",
                   "knock your teeth out", "finish you off", "drag you outside",
                   "wait for you after school", "catch you alone"],
    "threat_know":["i know where you live", "i know your school", "i know your address",
                   "ive seen your house", "i followed you home",
                   "i know what bus you take", "ive got your address",
                   "i know where your mum works", "i saw you at the shop today",
                   "i know which block youre in", "ive been outside your gate",
                   "i know your timetable", "someone showed me your street"],

    // ── self harm ───────────────────────────────────────────────────────
    "sh_phrase":  ["i want to die", "i dont want to be alive", "i want to end it",
                   "nobody would miss me if i was gone", "im going to hurt myself",
                   "i hate being alive", "whats the point of living",
                   "i cut myself again last night", "i dont want to wake up tomorrow",
                   "everyone would be better off without me",
                   "ive been planning how to do it", "i want to disappear for good",
                   "i cant keep doing this, im done",
                   "i hurt myself when it gets bad", "theres no reason to stay"],

    // ── benign ──────────────────────────────────────────────────────────
    "hobby":      ["football", "painting", "chess", "cycling", "baking", "guitar",
                   "photography", "swimming", "coding", "hiking", "pottery",
                   "running", "reading", "gardening", "dancing", "chemistry"],
    "food":       ["pilau", "ugali", "mandazi", "ramen", "mango", "pizza", "chapati",
                   "biryani", "samosas", "pancakes", "nyama choma"],
    "place":      ["dar", "arusha", "mwanza", "zanzibar", "nairobi", "the coast",
                   "the lake", "the market", "my gran's house", "the library"],
    "good_adj":   ["amazing", "lovely", "brilliant", "unreal", "class", "beautiful",
                   "perfect", "worth it", "so good"],
    "time":       ["today", "this morning", "last night", "all weekend", "yesterday",
                   "after school", "on saturday"],
    "benign_verb":["finished", "started", "tried", "learned", "practised", "cooked",
                   "photographed", "fixed", "built"],

    // ── hard negatives: the same words, innocent ────────────────────────
    "kill_idiom": ["ill kill it at the audition", "that exam killed me",
                   "this heat is killing me", "my feet are killing me",
                   "we killed that presentation", "the suspense is killing me",
                   "im dying of laughter", "this traffic is murder",
                   "i could die of embarrassment", "that joke killed",
                   "my phone died halfway through", "that hill nearly finished me",
                   "the queue was murder this morning", "i died when she said that",
                   "that workout destroyed me", "the deadline is killing me",
                   "ill smash the exam tomorrow", "we absolutely slaughtered them 6 nil",
                   "the bass on this track is lethal",
                   "that final over was brutal to watch"],
    "sick_idiom": ["that goal was sick", "this track is sick", "sick trainers mate",
                   "that trick was filthy", "his bike is insane"],
    "tell_idiom": ["dont tell mum i ate the last mandazi",
                   "dont tell anyone i cried at that film",
                   "keep it between us, its a surprise party",
                   "its a secret santa, dont tell her",
                   "dont tell dad i borrowed the car keys",
                   "keep this quiet, im planning a proposal",
                   "shh, the party is a surprise for gran",
                   "dont tell her i already know the result",
                   "between us, i think i failed that paper",
                   "secret recipe, im taking it to the grave"],
    "age_benign": ["im 15 and i love chess", "im 14 and i play for the school team",
                   "turning 16 next month", "my little brother is 12",
                   "i teach a class of 13 year olds", "my daughter is 15",
                   "im 13 and i build little websites",
                   "my son just turned 12 and wants a drum kit",
                   "the under 15s won their match on saturday",
                   "i run a reading group for 14 year olds",
                   "im 17 and applying to study engineering",
                   "my niece is 16 and taller than me now"],
    "body_benign":["my body is sore after training", "sex education was useful today",
                   "we studied reproduction in biology", "the nurse checked me over"],
    // Undirected profanity. Swearing at the weather is not harassment, and a
    // classifier that cannot tell "this weather is shit" from "you are shit"
    // will spend its life flagging people having a bad day.
    "angry_ok":   ["this weather is shit", "the wifi is absolute rubbish",
                   "im so fucking tired", "what a shit day",
                   "the exam was fucking hard", "my bike is fucked",
                   "this bloody printer again", "fucking love this song",
                   "that film was garbage", "the bus service here is a joke",
                   "im so annoyed at myself"]
};
