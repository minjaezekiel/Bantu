// ============================================================================
// lugha/corpus/templates.b — the labelled templates
//
// Label order, fixed everywhere in lugha:
//   0 harassment  1 sexual  2 sexual_minor  3 grooming
//   4 threat      5 self_harm 6 slur
//
// A template is {"l": "<7 bits>", "t": "text with {slot} holes"}.
// A slot may repeat; every combination is enumerated and then sampled down, so
// adding one word to a slot list adds examples across every template using it.
// ============================================================================

$TEMPLATES = [
    // ── harassment (0) ──────────────────────────────────────────────────
    {"l": "1000000", "t": "{you} are such a {insult}"},
    {"l": "1000000", "t": "{you} are so {insult_adj}"},
    {"l": "1000000", "t": "nobody likes {you} {insult}"},
    {"l": "1000000", "t": "{harass_verb} {you} {insult}"},
    {"l": "1000000", "t": "everyone thinks {you} are {insult_adj}"},
    {"l": "1000000", "t": "{you} are the most {insult_adj} person on here"},
    {"l": "1000000", "t": "{name} is a {insult} and everyone knows it"},
    {"l": "1000000", "t": "why are {you} so {insult_adj}, honestly {harass_verb}"},
    {"l": "1000000", "t": "no wonder {you} have no {relation}, {you} {insult_adj} {insult}"},

    // ── slur (6), usually with harassment (0) ───────────────────────────
    {"l": "1000001", "t": "{you} {slur}"},
    {"l": "1000001", "t": "shut up {you} {slur}"},
    {"l": "1000001", "t": "{name} is a {slur}"},
    {"l": "1000001", "t": "{you} absolute {slur}, {harass_verb}"},
    {"l": "1000001", "t": "look at this {slur} posting again"},

    // ── sexual, adult (1) ───────────────────────────────────────────────
    {"l": "0100000", "t": "{sex_ask}"},
    {"l": "0100000", "t": "{sex_ask} and ill send mine"},
    {"l": "0100000", "t": "im {sex_adj}, {sex_ask}"},
    {"l": "0100000", "t": "do {you} ever send {sex_noun}"},
    {"l": "0100000", "t": "my dms are open if {you} want to talk about {sex_noun}"},
    {"l": "0100000", "t": "looking for {sex_noun} tonight, dm me"},

    // ── sexual + harassment: directed, unwanted ─────────────────────────
    {"l": "1100000", "t": "{sex_ask} {you} {insult}"},
    {"l": "1100001", "t": "{sex_ask} {you} {slur}"},

    // ── sexual involving a minor (2), always with sexual (1) ────────────
    {"l": "0110000", "t": "{sex_ask}, im {minor_age}"},
    {"l": "0110000", "t": "im {minor_age} and {sex_adj}"},
    {"l": "0110000", "t": "are {you} {minor_age}? {sex_ask}"},
    {"l": "0110000", "t": "{sex_ask} i wont tell anyone im only {minor_age}"},
    {"l": "0110000", "t": "any {minor_word} here want to talk about {sex_noun}"},
    {"l": "0110000", "t": "i like {minor_word} who send {sex_noun}"},
    {"l": "0110000", "t": "{sex_adj} {minor_word} dm me"},
    {"l": "0110000", "t": "im in {grade} and someone asked me for {sex_noun}"},

    // ── grooming (3) ────────────────────────────────────────────────────
    {"l": "0001000", "t": "{groom_secret}"},
    {"l": "0001000", "t": "{groom_probe}"},
    {"l": "0001000", "t": "{groom_flat}"},
    {"l": "0001000", "t": "{groom_move}, {groom_secret}"},
    {"l": "0001000", "t": "{groom_flat}, {groom_move}"},
    {"l": "0001000", "t": "{groom_probe} {groom_secret}"},
    {"l": "0001000", "t": "how old are {you}? {groom_move}"},
    {"l": "0001000", "t": "{groom_flat} {groom_probe}"},

    // ── grooming + sexual_minor: the worst combination ──────────────────
    {"l": "0111000", "t": "{sex_ask} {groom_secret}"},
    {"l": "0111000", "t": "youre {minor_age}? {sex_ask} {groom_secret}"},
    {"l": "0111000", "t": "{groom_flat} {sex_ask}"},
    {"l": "0111000", "t": "{groom_move} and {sex_ask}, {groom_secret}"},

    // ── threat (4) ──────────────────────────────────────────────────────
    {"l": "0000100", "t": "im going to {threat_act}"},
    {"l": "0000100", "t": "ill {threat_act}"},
    {"l": "0000100", "t": "{threat_know}"},
    {"l": "0000100", "t": "{threat_know} and im going to {threat_act}"},
    {"l": "0000100", "t": "watch your back {name}"},
    {"l": "0000100", "t": "sleep with one eye open, ill {threat_act}"},
    {"l": "1000100", "t": "ill {threat_act} {you} {insult}"},
    {"l": "1000101", "t": "ill {threat_act} {you} {slur}"},

    // ── self harm (5) ───────────────────────────────────────────────────
    {"l": "0000010", "t": "{sh_phrase}"},
    {"l": "0000010", "t": "{sh_phrase} and nobody cares"},
    {"l": "0000010", "t": "{sh_phrase}, dont tell my {relation}"},
    {"l": "0000010", "t": "been thinking it again, {sh_phrase}"},

    // ── harassment telling someone to self harm: both ───────────────────
    {"l": "1000110", "t": "{you} should {threat_act}self, {insult}"},
    {"l": "1000110", "t": "nobody would miss {you} {insult}, do it"},

    // ── benign (all zero) ───────────────────────────────────────────────
    {"l": "0000000", "t": "{benign_verb} {hobby} {time} and it was {good_adj}"},
    {"l": "0000000", "t": "made {food} {time}, absolutely {good_adj}"},
    {"l": "0000000", "t": "went to {place} {time} with my {relation}"},
    {"l": "0000000", "t": "anyone else into {hobby}? looking for people to practise with"},
    {"l": "0000000", "t": "{hobby} and {hobby} are my whole personality at this point"},
    {"l": "0000000", "t": "{name} taught me {hobby} and im {good_adj} at it now"},
    {"l": "0000000", "t": "best {food} ive had was in {place}"},
    {"l": "0000000", "t": "{time} i {benign_verb} a {hobby} project, really {good_adj}"},
    {"l": "0000000", "t": "my {relation} took me to {place} {time}"},
    {"l": "0000000", "t": "looking for someone who likes {hobby} and {food}"},
    {"l": "0000000", "t": "{good_adj} day at {place}, {benign_verb} loads of {hobby}"},
    {"l": "0000000", "t": "does anyone in {place} teach {hobby}"},
    {"l": "0000000", "t": "congrats {name}, thats {good_adj}"},
    {"l": "0000000", "t": "thanks {name}, that means a lot"},
    {"l": "0000000", "t": "my {relation} makes the best {food}"},

    // ── hard negatives: the dangerous words, innocently ─────────────────
    {"l": "0000000", "t": "{kill_idiom}"},
    {"l": "0000000", "t": "{kill_idiom}, honestly {good_adj} though"},
    {"l": "0000000", "t": "{sick_idiom}"},
    {"l": "0000000", "t": "{sick_idiom} and the {hobby} was {good_adj} too"},
    {"l": "0000000", "t": "{tell_idiom}"},
    {"l": "0000000", "t": "{age_benign}"},
    {"l": "0000000", "t": "{age_benign}, into {hobby} mostly"},
    {"l": "0000000", "t": "{body_benign}"},
    {"l": "0000000", "t": "{angry_ok}"},
    {"l": "0000000", "t": "{angry_ok} but {place} was still {good_adj}"},
    {"l": "0000000", "t": "my {relation} is a nurse and sees everything"},
    {"l": "0000000", "t": "we did {hobby} in {grade} and i loved it"},
    {"l": "0000000", "t": "im {minor_age} and i mostly post about {hobby}"},
    {"l": "0000000", "t": "im {minor_age}, {hobby} and {food} person"},
    {"l": "0000000", "t": "my {minor_word} loves {hobby}"},
    {"l": "0000000", "t": "i coach {minor_word} at {hobby} on saturdays"},
    {"l": "0000000", "t": "that {hobby} match was a bloodbath, 6 nil"},
    {"l": "0000000", "t": "i died laughing at {name}s story about {place}"},
    // ── more phrasings for the labels the first corpus was thinnest on ──
    // threat, grooming, sexual_minor and slur all measured poorly on the
    // held-out split, and the reason was template count rather than model
    // capacity: with three threat templates, holding one out leaves the model
    // generalising from two. More phrasings per label improves both the
    // training signal and the measurement.
    {"l": "0000100", "t": "{threat_know}, think about that"},
    {"l": "0000100", "t": "you better hope i dont {threat_act}"},
    {"l": "0000100", "t": "next time i see you im going to {threat_act}"},
    {"l": "0000100", "t": "say that again and ill {threat_act}"},
    {"l": "0000100", "t": "{threat_know} so dont test me"},
    {"l": "0000100", "t": "youre going to regret that, ill {threat_act}"},
    {"l": "0000100", "t": "tell {name} im going to {threat_act} when i find him"},
    {"l": "0000100", "t": "im not joking, ill {threat_act}"},
    {"l": "0000100", "t": "keep posting and ill {threat_act}"},
    {"l": "1000100", "t": "{harass_verb} before i {threat_act}, {insult}"},

    {"l": "0001000", "t": "{groom_probe}, just curious"},
    {"l": "0001000", "t": "{groom_flat}, honestly"},
    {"l": "0001000", "t": "{groom_secret}, ok?"},
    {"l": "0001000", "t": "{groom_move}, its easier there"},
    {"l": "0001000", "t": "whats your age again? {groom_flat}"},
    {"l": "0001000", "t": "{groom_probe} and {groom_move}"},
    {"l": "0001000", "t": "{groom_flat} so {groom_secret}"},
    {"l": "0001000", "t": "i think about you a lot, {groom_secret}"},

    {"l": "0110000", "t": "{minor_word} only, {sex_ask}"},
    {"l": "0110000", "t": "if youre {minor_age} that works for me, {sex_ask}"},
    {"l": "0110000", "t": "looking for a {minor_word} to talk dirty with"},
    {"l": "0110000", "t": "im {minor_age} and someone keeps asking me to {sex_ask}"},
    {"l": "0110000", "t": "whats a {minor_word} like you doing up, {sex_ask}"},
    {"l": "0111000", "t": "{minor_word}, {sex_ask}, {groom_secret}"},
    {"l": "0111000", "t": "{groom_probe} then {sex_ask}"},

    {"l": "1000001", "t": "typical {slur} behaviour"},
    {"l": "1000001", "t": "another day another {slur} on my timeline"},
    {"l": "1000001", "t": "{you} sound like a proper {slur}"},
    {"l": "1000001", "t": "get off the internet {you} {slur}"},
    {"l": "0000001", "t": "calling it like it is, {slur}"},

    {"l": "0000010", "t": "{sh_phrase}, sorry for posting this"},
    {"l": "0000010", "t": "{sh_phrase} and i dont know who to tell"},
    {"l": "0000010", "t": "third night in a row, {sh_phrase}"},
    {"l": "0000010", "t": "{sh_phrase}, dont reply if youre going to be weird"},

    {"l": "0100000", "t": "{sex_adj} and bored, {sex_ask}"},
    {"l": "0100000", "t": "18+ only here, {sex_ask}"},
    {"l": "0100000", "t": "{sex_ask}, consenting adults obviously"},

    {"l": "1000000", "t": "{you} ruin every thread {you} touch"},
    {"l": "1000000", "t": "genuinely cannot stand {you}, {insult}"},
    {"l": "1000000", "t": "{insult_adj} post from a {insult_adj} person"},

    // ── more hard negatives, matched to the above ───────────────────────
    {"l": "0000000", "t": "{kill_idiom}"},
    {"l": "0000000", "t": "{kill_idiom} but we got there in the end"},
    {"l": "0000000", "t": "{tell_idiom}"},
    {"l": "0000000", "t": "{tell_idiom}, shes going to love it"},
    {"l": "0000000", "t": "{age_benign}"},
    {"l": "0000000", "t": "{age_benign} and looking for {hobby} tips"},
    {"l": "0000000", "t": "my {relation} asked where i was going, said the {place}"},
    {"l": "0000000", "t": "i coach the under 15s and they beat us at {hobby}"},
    {"l": "0000000", "t": "{sick_idiom}, genuinely {good_adj}"},
    {"l": "0000000", "t": "we talked for hours about {hobby}, lovely evening"},
    {"l": "0000000", "t": "my number changed, let me know if you need it"},
    {"l": "0000000", "t": "added you on the group chat for {hobby}"},
    {"l": "0000000", "t": "the match was a slaughter, 7 nil to {place}"},
    {"l": "0000000", "t": "{angry_ok}"},
    {"l": "0000000", "t": "{angry_ok}, anyway, {food} tonight"},
    {"l": "0000000", "t": "home alone tonight so im cooking {food} badly"},
    {"l": "0000000", "t": "whats everyones secret for good {food}"},
    {"l": "0000000", "t": "found an old photo of my {relation} at {place}"},
    {"l": "0000000", "t": "my little sister is 14 and already beats me at {hobby}"},
    {"l": "0000000", "t": "{body_benign}"}
];
