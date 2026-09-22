"""
build_embeddings.py
--------------------
Builds Polish word -> embedding-vector dictionaries and saves them to .npz
files.

Running this script with no arguments produces four files:
    builtin_embeddings.npz              (524 most common words, normalized)
    builtin_embeddings_unnormalize.npz  (524 most common words, raw vectors)
    polish_embeddings.npz               (10,000 most common nouns, normalized)
    polish_embeddings_unnormalize.npz   (10,000 most common nouns, raw vectors)

The "normalized" files contain L2-normalized vectors (cosine similarity =
dot product); the "unnormalize" files contain the model's raw output
vectors.

Requirements:
    pip install sentence-transformers numpy

Usage:
    python build_embeddings.py
    python build_embeddings.py --subtlex SUBTLEX_PL.tsv --top 20000
    python build_embeddings.py --words-file my_words.txt --output my_embeddings

Downloading SUBTLEX-PL (~100k words):
    https://osf.io/5a76z/  ->  "Files" tab -> download the file
    Place it next to this script (default expected name: subtlex-pl.csv),
    or point to it with --subtlex.
"""
import csv
import argparse

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SIZE = 256
DEFAULT_SUBTLEX_PATH = "subtlex-pl.csv"
DEFAULT_TOP_N = 10000

# ── Built-in list of the ~524 most common Polish words ───────────────────────
BUILTIN_WORDS = [
    # nouns - everyday environment
    "dom","czas","człowiek","rok","dzień","kraj","praca","miejsce","świat","życie","sprawa","sposób","część",
    "miasto","oko","słowo","noc","głowa","ręka","woda","ogień","ziemia","niebo","morze","las","droga","ulica",
    "okno","drzwi","stół","krzesło","łóżko","samochód","pociąg","autobus","szkoła","szpital","sklep","kościół",
    "park","mieszkanie","pokój","kuchnia","łazienka","ogród","podwórko","balkon","garaż","piwnica","strych",
    # people and relationships
    "mama","tata","brat","siostra","syn","córka","mąż","żona","przyjaciel","kolega","dziecko","rodzina",
    "miłość","znajomy","sąsiad","szef","pracownik","uczeń","nauczyciel","lekarz","pacjent","klient","gość",
    # abstract nouns
    "pieniądze","zdrowie","szczęście","problem","pytanie","odpowiedź","historia","kultura","sztuka",
    "muzyka","film","książka","gazeta","telefon","komputer","internet","wiadomość","informacja","prawda",
    "kłamstwo","nauka","badanie","teoria","wynik","metoda","analiza","dane","technologia","fizyka","chemia",
    "biologia","matematyka","informatyka","medycyna","biznes","firma","rynek","produkt","usługa","zysk",
    "bank","kredyt","podatek","budżet","gospodarka","ekonomia","handel","sport","mecz","gra","zawodnik",
    "drużyna","trener","mistrzostwo","pogoda","temperatura","wiatr","deszcz","śnieg","słońce","burza",
    # proper nouns / geography
    "Polska","Warszawa","Kraków","Wrocław","Poznań","Gdańsk","Łódź","Katowice","Lublin","Białystok",
    "Europa","Niemcy","Francja","Rosja","Anglia","Ameryka","Chiny","Japonia","Włochy","Hiszpania",
    # religion / philosophy
    "Bóg","kościół","wiara","modlitwa","dusza","grzech","łaska","zbawienie","anioł","diabeł",
    # politics and society
    "rząd","prezydent","minister","parlament","prawo","sąd","policja","wojsko","armia","żołnierz",
    "partia","wybory","głosowanie","kampania","reforma","zmiana","rewolucja","demokracja","społeczeństwo",
    "klasa","grupa","wspólnota","organizacja","stowarzyszenie","instytucja","urząd","obywatel","naród",
    # time and space
    "przeszłość","teraźniejszość","przyszłość","epoka","era","wiek","okres","etap","chwila","sekunda",
    "minuta","godzina","tydzień","miesiąc","kwartał","przestrzeń","odległość","kierunek","strona",
    "centrum","obrzeże","granica","terytorium","obszar","region","kontynent","wyspa","rzeka","góra",
    # emotions and psychology
    "nadzieja","marzenie","pragnienie","ambicja","motywacja","determinacja","wytrwałość","odwaga",
    "strach","niepokój","stres","depresja","ból","cierpienie","żal","smutek","radość","entuzjazm",
    "złość","zazdrość","wstyd","duma","miłość","nienawiść","ciekawość","nuda","zdziwienie","rozczarowanie",
    # food
    "jedzenie","chleb","mięso","ryba","warzywo","owoc","zupa","obiad","śniadanie","kolacja","deser",
    "kawa","herbata","wino","piwo","wódka","mleko","ser","masło","jajko","cukier","sól","pieprz","mąka",
    "ryż","makaron","ziemniak","pomidor","ogórek","cebula","czosnek","marchew","kapusta","sałata",
    # clothing
    "ubranie","sukienka","spodnie","koszula","buty","kurtka","czapka","szalik","rękawiczki","płaszcz",
    "garnitur","sweter","bluza","bielizna","skarpetki","pasek","torebka","plecak","parasol",
    # education and science
    "edukacja","przedszkole","liceum","uczelnia","wydział","kierunek","egzamin","dyplom","stopień",
    "nagroda","kara","zasada","norma","wartość","etyka","moralność","sprawiedliwość","wolność","równość",
    # character and traits
    "charakter","osobowość","zachowanie","postawa","opinia","pogląd","przekonanie","ideał","cel",
    "plan","projekt","pomysł","koncepcja","strategia","decyzja","wybór","rozwiązanie","kompromis",
    "sukces","porażka","postęp","rozwój","innowacja","odkrycie","wynalazek","przełom","osiągnięcie",
    "wyzwanie","trudność","przeszkoda","zagrożenie","ryzyko","niebezpieczeństwo","kryzys","katastrofa",
    # nature
    "drzewo","kwiat","trawa","liść","gałąź","korzeń","pień","kora","owoc","nasiono","łąka","pole",
    "rzeka","jezioro","ocean","fala","brzeg","piasek","skała","kamień","gleba","błoto","piasek",
    "zwierzę","pies","kot","koń","krowa","świnia","kura","ryba","ptak","owad","mucha","pszczoła",
    "orzeł","wróbel","gołąb","wilk","lis","niedźwiedź","zając","jeleń","sarna","dzik","wiewiórka",
    # home and furniture
    "meble","sofa","fotel","szafa","komoda","biurko","półka","lustro","lampa","dywan","zasłona",
    "pościel","poduszka","kołdra","ręcznik","mydło","szampon","pasta","szczotka","grzebień",
    # technology
    "maszyna","silnik","prąd","bateria","kabel","wtyczka","ekran","klawiatura","mysz","drukarka",
    "aparat","kamera","telewizor","radio","głośnik","słuchawki","mikrofon","antena","sieć","serwer",
    # medicine and body
    "ciało","głowa","twarz","nos","usta","ucho","szyja","ramię","łokieć","nadgarstek","palec","paznokieć",
    "klatka","brzuch","plecy","biodro","kolano","kostka","stopa","serce","płuco","wątroba","nerka",
    "mózg","kość","mięsień","skóra","krew","nerw","choroba","lek","leczenie","operacja","diagnoza",
    "symptom","epidemia","wirus","bakteria","szczepionka","terapia","rehabilitacja","profilaktyka",
    # literature and culture
    "literatura","powieść","opowiadanie","wiersz","dramat","bajka","legenda","mit","biografia","esej",
    "język","zdanie","tekst","litera","gramatyka","ortografia","znaczenie","narracja","dialog","opis",
    "tradycja","zwyczaj","obyczaj","rytuał","ceremonia","święto","festiwal","koncert","spektakl","wystawa",
    # transport and infrastructure
    "transport","lotnisko","dworzec","stacja","port","most","tunel","autostrada","skrzyżowanie","chodnik",
    "parking","paliwo","silnik","koło","kierownica","hamulec","silnik","motocykl","rower","hulajnoga",
]


def deduplicate(words: list) -> list:
    seen = set()
    result = []
    for w in words:
        if w not in seen:
            seen.add(w)
            result.append(w)
    return result


def load_subtlex_nouns(path: str, top_n: int) -> list:
    """Loads the top_n most frequent nouns from a SUBTLEX-PL frequency file."""
    words = []
    seen = set()
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            pos = row.get("all.pos", "").strip().lower()
            if pos != ".subst.":
                continue
            word = row["spelling"].strip()
            if word not in seen:
                seen.add(word)
                words.append(word)
            if len(words) >= top_n:
                break
    return deduplicate(words)


def load_words_file(path: str, top_n: int | None) -> list:
    with open(path, encoding="utf-8") as f:
        words = [line.strip() for line in f if line.strip()]
    if top_n:
        words = words[:top_n]
    return deduplicate(words)


def encode(model: SentenceTransformer, words: list) -> np.ndarray:
    """Encodes words once and returns raw (non-normalized) vectors."""
    print(f"  Encoding {len(words)} words...")
    return model.encode(
        words,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )


def save(output_path: str, words: list, embeddings: np.ndarray) -> None:
    np.savez_compressed(
        output_path,
        words=np.array(words, dtype=object),
        embeddings=embeddings,
    )
    dim = embeddings.shape[1]
    size_mb = embeddings.nbytes / 1024 / 1024
    print(f"  \u2713 Saved '{output_path}.npz' "
          f"(dim={dim}, size={size_mb:.1f} MB, words={len(words)})")


def build_pair(model: SentenceTransformer, output_prefix: str, words: list) -> None:
    """Encodes a word list once, then saves both a normalized and an
    unnormalized .npz file from that single encoding pass."""
    print(f"\n[{output_prefix}]")
    raw = encode(model, words)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    normalized = raw / norms

    save(output_prefix, words, normalized)
    save(f"{output_prefix}_unnormalize", words, raw)


def main():
    parser = argparse.ArgumentParser(
        description="Builds Polish word embedding dictionaries.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python build_embeddings.py
  python build_embeddings.py --subtlex SUBTLEX_PL.tsv --top 20000
  python build_embeddings.py --words-file my_words.txt --output my_embeddings""",
    )
    parser.add_argument("--subtlex", default=DEFAULT_SUBTLEX_PATH,
                         help=f"SUBTLEX-PL frequency file (.csv/.tsv), default: {DEFAULT_SUBTLEX_PATH}")
    parser.add_argument("--top", type=int, default=DEFAULT_TOP_N,
                         help=f"Number of most frequent nouns to take from --subtlex, default: {DEFAULT_TOP_N}")
    parser.add_argument("--words-file",
                         help="Optional custom word list (one word per line). "
                              "If given, this replaces the SUBTLEX-based list and is built "
                              "as a single pair using --output instead of 'polish_embeddings'.")
    parser.add_argument("--output", default="polish_embeddings",
                         help="Output name prefix used together with --words-file")
    args = parser.parse_args()

    model = SentenceTransformer(MODEL_NAME)

    # 1) Built-in word list -> builtin_embeddings.npz / builtin_embeddings_unnormalize.npz
    build_pair(model, "builtin_embeddings", deduplicate(BUILTIN_WORDS))

    # 2) Either a custom word list, or the top-N SUBTLEX-PL nouns
    if args.words_file:
        custom_words = load_words_file(args.words_file, args.top)
        build_pair(model, args.output, custom_words)
    else:
        polish_words = load_subtlex_nouns(args.subtlex, args.top)
        build_pair(model, "polish_embeddings", polish_words)

    print("\nDone.")


if __name__ == "__main__":
    main()