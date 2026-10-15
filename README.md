# Racet – AI-träning man kan se

Bilar som lär sig köra. En liten språkmodell, **all-MiniLM-L6-v2** (eller **Laya**), läser
vad varje bil ser som text och väljer manöver, och tränas på grafikkortet mellan
generationerna. Ett pedagogiskt exempel på hur man tränar en modell i praktiken: best
practice, krascher som straff, temperatur, och tre banor som visar om modellen har lärt sig
köra eller bara memorerat. Designen följer Laya Training Visualization Tool (4Front 2 Market):
mörk ram, ljus spelyta, instrument för grafikkortet.

Repot har också fyra turbaserade spel där en beslutsmodell gör varje drag – se *Spelen*.

```bash
python play.py
```

Öppnar spelen på http://localhost:8765 och racet på http://localhost:8765/race.

**Kom igång.** Racet och Laya kräver ett NVIDIA-grafikkort och en Python med PyTorch (CUDA)
och transformers. Modellerna läses från Hugging Face-cachen, så ladda ned dem en gång först:

```bash
pip install torch transformers huggingface_hub --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple
```

```bash
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2
```

```bash
huggingface-cli download convaiinnovations/laya
```

Nimble i de turbaserade spelen körs via [Ollama](https://ollama.com)
(`hf.co/ggml-org/Bespoke-Nimble-9B-v3-GGUF:Q8_0`). Utan PyTorch fungerar Nimble, men inte
racet eller Laya.

## Spelen

| Flik | `--game` | Modellen väljer | Lär sig |
|---|---|---|---|
| Bilspelet – väj för bilar på en trefilig väg | `car` | vänster, rakt fram, höger | nej |
| Bilspelet – självlärande | `learn` | vänster, rakt fram, höger | vilka drag som krockar i vilket läge |
| Isvaken – självlärande | `lake` | upp, ner, vänster, höger | en karta över sjön |
| Snake | `snake` | upp, ner, vänster, höger | nej |
| Racet – 100 bilar (`/race`) | – | sex manövrer | Laya tränas mellan generationerna |

## Modellerna

| Modell | Vad det är | Körs | Licens |
|---|---|---|---|
| `nimble` (förvald) | `hf.co/ggml-org/Bespoke-Nimble-9B-v3-GGUF:Q8_0` | Ollama, `/v1/systemone` | CC BY-NC 4.0, ej kommersiellt |
| `laya` | `convaiinnovations/laya`, ModernBERT-large med beslutshuvud | PyTorch på grafikkortet, från Hugging Face-cachen | Apache-2.0 |

Nimble hämtas och får sitt korta namn med:

```bash
ollama pull hf.co/ggml-org/Bespoke-Nimble-9B-v3-GGUF:Q8_0
```

```bash
ollama cp hf.co/ggml-org/Bespoke-Nimble-9B-v3-GGUF:Q8_0 nimble
```

I de turbaserade spelen spelar Nimble bra (Snake 29–30 äpplen, bilspelet 1000 drag utan
krock). Laya otränad spelar nära slump där (krock efter 8 drag), som Layas egen
dokumentation säger om grundmodellen. I racet tränas Laya och lär sig.

## Racet – 100 bilar

Två förare, valbara överst, som båda tränas på grafikkortet mellan generationerna. Bara den
valda ligger på grafikkortet: byter man förare släpps den andra, och det den lärt sig läses in
från sin fil igen när man byter tillbaka.

| Förare | Modell | Vad som tränas |
|---|---|---|
| **Laya** | `convaiinnovations/laya`, 421M parametrar, förtränad för beslut | bara beslutshuvudet; den stora kodaren är fryst |
| **all-MiniLM-L6-v2** (förvald) | `sentence-transformers/all-MiniLM-L6-v2`, en väldigt liten BERT, 22M parametrar, kan inget om racet från början | hela modellen, plus ett nytt beslutslager |

En bil beskriver vad den ser i text – *"Speed: fast. The road ahead bends right. Distance to
the road edge: far left close, …"* – och modellen väljer en av sex manövrer: skarpt vänster,
vänster, rakt och gasa, höger, skarpt höger, rakt och bromsa.

**Del 1 · runt banan.**
1. Första generationen väljer alla bilar själva och kraschar. Den som kom längst blir
   **best practice** – visas med två decimaler, t.ex. *11,59 % av banan*.
2. Därefter kör alla 100 exakt som best practice fram till 0,5 s före dess krasch (lila
   linje på banan). Där tar modellen över och bilarna gör lite olika val: de chansar lite
   till mycket, det vill säga kör med temperatur 0,2–1,0. En av tio beslut är dessutom helt
   slumpat (epsilon-greedy).
3. Den som kommer längst blir ny best practice. Slås inget rekord flyttas punkten där bilarna börjar köra själva
   0,5 s längre bakåt per generation, högst 3 s.

**Del 2 · kapa tiden.** När första bilen kommer runt får varje bil en egen
punkt där den börjar köra själv, utspridd över hela rekordvarvet (lila prickar). Bilarna provar nya linjer
på många ställen samtidigt, och några kraschar på vägen. Med all-MiniLM-L6-v2 gick
varvtiden från 34,5 s till 15,5 s på en generation och till 14,5 s i generation 24.

**Provbilen** (turkos) kör med temperatur 0 – alltid det modellen tror mest på – och följer
aldrig best practice. Den kör på modellen ensam från start, så dess kurva (*bara modellen*)
visar vad modellen själv har lärt sig. Övriga kurvor får hjälp av best practice.

**Bana: lärt sig eller memorerat?** Styrkortet har tre banor: *Träning*, *Spegelvänd*
(vänsterkurvor blir högerkurvor) och *Ny bana*. Modellen följer med mellan banorna, men best
practice och krascher hör till banan. Best practice är memorerade rattrörelser och fungerar
bara på sin bana; det modellen lärt sig om fart, kurvor och vägkanter fungerar överallt.
Uppmätt med all-MiniLM-L6-v2 efter 7 generationer på träningsbanan: provbilen kom 47 % på
den spegelvända och 38 % på den nya banan på första försöket, och i båda fallen kom någon av
de chansande bilarna runt hela varvet redan i första generationen.

**Sensorer av** gör bilarna blinda: de vet bara sin egen fart, inte var vägkanten är eller
hur vägen svänger. Gäller från nästa beslut.

**Antal bilar** väljs i styrkortet: 10, 100, 1 000 eller 10 000. Ett byte startar om den
pågående generationen. Uppmätt tid per simuleringssteg i webbläsaren, när alla bilar läser
av sina sensorer (värsta fallet):

| Bilar | Per steg | Högsta fart |
|---|---|---|
| 100 | 0,15 ms | ~110× realtid |
| 1 000 | 0,74 ms | ~22× |
| 10 000 | 6,1 ms | ~2,7× |

Taket är webbläsarens processor, inte modellerna: bilar med samma läge delar svar, så en
generation har några hundra till ett par tusen olika lägen oavsett antal bilar. Träningen
tar högst 3 000 lägen per generation, och Laya sparar högst 6 000 lägen på grafikkortet
(runt 1,1 GB). Med all-MiniLM-L6-v2 och 10 000 bilar tog en generation 7–70 s; tiden
går till simuleringen, modellen och träningen tar några sekunder.

**Träningen.** Mellan generationerna: val som ledde till krasch inom åtta beslut straffas,
val som tog bilen framåt förstärks, och modellen härmar best practice och generationens tio
bästa. Alla krascher syns som röda kryss på banan.

Uppmätt, 30–40 generationer per förare:

| | Runt hela banan | Varvtid sedan |
|---|---|---|
| all-MiniLM-L6-v2 | generation 3 (52,5 s) | 16,6 s i generation 25 |
| Laya | generation 13 (49,7 s) | 46,3 s i generation 40 |

Nimble provades också som förare: 11,87 % redan i första generationen, men utan träning
stannade den vid 21 %. Den är kvar i de turbaserade spelen, där den spelar bra.

Det förarna lärt sig sparas i `memory/race_laya.pt` och `memory/race_minibert.pt`, och
racets rekord per bana i webbläsaren. **Börja om** nollställer föraren och dess rekord på alla banor.
**Mellanslag** spelar och pausar.

## Grafikkortet

Instrumenten visar belastning, grafikminne, tokens/s och effekt. Värdena kommer från
`nvidia-smi` via spelservern, en gång per sekund. Tokens/s gäller den tid modellen
faktiskt räknar.

Laya på processorn i stället för grafikkortet: uppmätt 198 ms per beslut mot 25 ms, och
13 s mot 0,16 s för 100 beslut på en gång. Det fungerar, men träningen mellan
generationerna blir minuter i stället för sekunder.

## Självinlärningen i spelen

Modellernas vikter ändras inte i `learn` och `lake`; det som växer är ett minne som
skrivs in i lägesbeskrivningen inför varje drag. Det sparas per spel och modell i
`memory/`, och **Glöm allt** nollställer det.

- **Bilspelet – självlärande** minns läge → drag → hur många gånger det gick bra eller
  slutade i krock.
- **Isvaken – självlärande** minns en karta: vakar, säker is, och hur långt det verkar
  vara till skatten från varje ruta. En ruta räknas som minst ett steg längre bort än sin
  bästa granne, så återvändsgränder blir dyrare tills en annan väg ser bättre ut (LRTA\*).

## Utan webbläsare

```bash
python play.py --game lake --headless 25
```

Spelar 25 omgångar i terminalen och skriver ut resultat och tokens/s.

## Filer

- `play.py` – startar allt; här står listan över spel och modeller.
- `engine.py` – Nimble via Ollama, Laya-tjänsten, minnet, webbservern, GPU-status och terminalläget.
- `laya_model.py` – Layas modell, laddad från Hugging Face-cachen (från convaiinnovations/laya, Apache-2.0).
- `race_text.py` – lägestexten och manövrerna i racet.
- `race_train.py` – träningen mellan generationerna, gemensam för förarna.
- `race_laya.py`, `race_bert.py` – förarna Laya och den lilla BERT:en.
- `car.py`, `car_learn.py`, `lake.py`, `snake.py` – ett spel var.
- `index.html`, `race.html` – sidorna; `theme.css` – gemensam design; `gpu.js` – GPU-status och instrument.
- `memory/` – det de självlärande spelen och racet har lärt sig (skapas vid körning, finns inte i repot).

Ollama på annan adress: sätt miljövariabeln `OLLAMA_URL`.
