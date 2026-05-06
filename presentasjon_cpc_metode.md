# 🔍 Automatisk CPC-klassifisering av patenter

> **Prosjekt:** MCP Patent Classification System  
> **Dato:** Mai 2026  
> **Formål:** Presentasjon — ikke-teknisk oversikt

---

## 1. Hva er dette?

Vi har bygget et **AI-verktøy som automatisk foreslår CPC-koder** (Cooperative Patent Classification) for patentsøknader.

I stedet for å manuelt lete gjennom tusenvis av CPC-kategorier for å finne riktig kode, kan en patentgransker nå:

1. **Lime inn patentteksten** i et nettleserbasert skjema
2. **Klikke én knapp**
3. **Få tilbake en rangert liste** med de mest relevante CPC-kodene — med begrunnelse

> 💡 **Tenk på det slik:** Systemet fungerer som en erfaren kollega som leser patentsøknaden, identifiserer de tekniske kjernebegrepene, og foreslår hvilke CPC-koder som passer best — alt på under ett minutt.

---

## 2. Hvorfor trenger vi dette?

### Utfordringen med manuell klassifisering

| Problem | Konsekvens |
|---------|-----------|
| CPC-systemet har **over 260 000 koder** | Vanskelig å ha full oversikt |
| Nye teknologier krysser tradisjonelle grenser | Feil klassifisering → patent blir vanskelig å finne |
| Klassifisering er tidkrevende | Tar ofte 15–30 minutter per søknad |
| Konsistens varierer mellom granskere | Samme oppfinnelse kan få ulike koder |

### Hva verktøyet løser

- ⏱️ **Raskere:** Fra 15–30 min til under 1 minutt for et førsteutkast
- 🎯 **Mer presist:** Bruker offisielle EPO-definisjoner, ikke hukommelse
- 🔄 **Konsistent:** Samme tekst gir alltid samme resultat
- 📚 **Oppdatert:** Basert på CPC-skjemafilene fra 2026

---

## 3. Hvordan fungerer det? (Forenklet)

Systemet bruker en **flerfase-tilnærming** — akkurat som en erfaren gransker ville gjort det:

### 📋 Oversikt over prosessen

```
    📝 Patenttekst                    🏷️ CPC-koder
    (input fra bruker)                (forslag til gransker)
          │                                  ▲
          ▼                                  │
    ┌─────────────┐    ┌──────────────┐    ┌─────────────┐
    │  FASE 1     │───▶│  FASE 2–3    │───▶│  FASE 4–7   │
    │  Forstå     │    │  Søk og      │    │  Verifiser   │
    │  teksten    │    │  rangere     │    │  og velg     │
    └─────────────┘    └──────────────┘    └─────────────┘
```

---

### Fase 1 — «Forstå patentet»

Systemet leser patentteksten og trekker ut:

- 🔧 **Hva er oppfinnelsen?** (teknisk objekt)
- ❓ **Hvilket problem løser den?**
- ⚙️ **Hva er kjernefunksjonen?** (f.eks. «tetting av brønnhull», «gjenkjenning av tale»)
- 🏭 **I hvilket domene brukes den?** (f.eks. olje/gass, medisin, telekom)
- 📝 **Viktige tekniske termer** — rangert etter relevans

> **Analogi:** Dette tilsvarer at en gransker leser sammendraget og kravene, og noterer stikkord og fagområde.

---

### Fase 2–3 — «Søk og rangere»

Med informasjonen fra Fase 1 søker systemet gjennom **hele CPC-hierarkiet**:

1. **Identifiserer relevante CPC-klasser** (f.eks. E21B for brønnboring, G06N for maskinlæring)
2. **Utvider til undergrupper** ved å lese de offisielle XML-definisjonene fra EPO
3. **Scorer hver undergruppe** basert på:
   - Hvor godt tittelen matcher de tekniske termene
   - Om domenet stemmer overens
   - Om funksjonsbeskrivelsen passer

> **Analogi:** Som å slå opp i CPC-tabellverket og sammenligne definisjonene med oppfinnelsens kjernebegreper — men gjort på sekunder i stedet for minutter.

---

### Fase 4–7 — «Kvalitetskontroll»

Systemet kjører deretter en **valideringsrunde**:

- ✅ Sjekker at koden passer med oppfinnelsens **funksjon** (ikke bare utseende)
- ✅ Verifiserer at **domenet** er riktig (f.eks. ikke foreslå medisin-koder for en mekanisk oppfinnelse)
- ✅ Fjerner koder som er **for generelle** eller **misvisende**
- ✅ Velger én **anbefalt «best-code»** med begrunnelse

> **Analogi:** En seniorgransker som dobbeltsjekker arbeidet og sikrer at valgt kode faktisk dekker oppfinnelsens essens.

---

## 3b. Kunnskapsgrafen/Knowledge Graph — hjernen bak søket

En nøkkelkomponent i systemet er **CPC-kunnskapsgrafen** — en intelligent, strukturert database som gjør det mulig å søke etter CPC-koder basert på **mening**, ikke bare nøkkelord.

### Hva er en kunnskapsgraf?

Tenk på det som et **digitalt kart over hele CPC-systemet**:

```
                    ┌──────────┐
                    │   G06    │  ← Seksjon (Fysikk / Databehandling)
                    └────┬─────┘
                         │
              ┌──────────┼──────────┐
              │          │          │
         ┌────▼───┐ ┌────▼───┐ ┌───▼────┐
         │ G06F   │ │ G06N   │ │ G06Q   │  ← Klasser
         │ Data-  │ │ Maskin-│ │ Forr.- │
         │ beh.   │ │ læring │ │ metoder│
         └────┬───┘ └────┬───┘ └────────┘
              │          │
         ┌────▼────┐ ┌───▼─────┐
         │G06F16/33│ │G06N3/08 │  ← Undergrupper
         │Søke-   │ │Nevrale  │
         │systemer │ │nett     │
         └─────────┘ └─────────┘
```

Hver kode er koblet til sine **overordnede** og **underordnede** koder, akkurat som et familietre. Systemet kan navigere opp og ned i dette treet for å finne den mest presise koden.

### Semantisk søk — forstå mening, ikke bare ord

Det som gjør kunnskapsgrafen spesielt kraftig, er **semantisk søk med AI-embeddings**:

| Tradisjonelt søk | Semantisk søk (vårt system) |
|-------------------|-----------------------------|
| Søker etter *nøyaktig* ordmatch | Forstår *betydningen* av teksten |
| "chatbot" finner bare koder med ordet "chatbot" | "chatbot" finner også "dialogsystem", "samtalebasert AI", "talegjenkjenning" |
| Mye relevant faller utenfor | Fanger opp synonymer og relaterte konsepter |

**Hvordan fungerer det i praksis?**

1. 📚 Systemet har allerede konvertert **alle 260 000+ CPC-titler** til matematiske vektorer ("embeddings") — en slags AI-fingeravtrykk av meningen i hver tittel
2. 🔍 Når brukeren sender inn patenttekst, konverteres også den til et fingeravtrykk
3. 📐 Systemet sammenligner fingeravtrykkene og finner de CPC-kodene som har **mest lignende mening** (kosinuslikhet)
4. 🎯 Resultatet er koder som er semantisk relevante — selv om de bruker helt andre ord

> **Analogi:** Tenk deg at du kan beskrive en oppfinnelse med dine egne ord, og systemet forstår hva du mener — selv om EPO bruker helt annen terminologi i sine definisjoner.

### Hybrid-tilnærming (60/40)

Systemet bruker en smart kombinasjon:
- **60 %** vekt på semantisk søk i hele patentteksten
- **40 %** vekt på matching av spesifikke tekniske nøkkeltermer

Dette sikrer at både den **brede konteksten** og de **spesifikke detaljene** teller.

---

## 4. Hva ser brukeren?

Brukeren åpner et **nettleserbasert skjema** (Streamlit-applikasjon) som ser slik ut:

### Steg 1: Lim inn patenttekst
Brukeren limer inn beskrivelse og eventuelt patentkrav i to tekstfelt.

### Steg 2: Klikk «Klassifiser»
Systemet behandler teksten (ca. 30–60 sekunder).

### Steg 3: Se resultater
Brukeren får:

| Resultat | Beskrivelse |
|----------|-------------|
| 🎯 **Teknisk objekt** | Hva oppfinnelsen handler om |
| ⚙️ **Kjernefunksjon** | Hva oppfinnelsen *gjør* |
| 🏭 **System/domene** | Hvilket teknisk felt den tilhører |
| 📊 **Rangerte CPC-koder** | Topp 7 forslag med score og begrunnelse |
| 🏆 **Beste kode** | Én anbefalt kode med konfidensgrad |
| 📋 **Tekniske termer** | Rangert liste over viktige begreper fra teksten |

---

## 5. Systemets tre hoveddeler

Systemet er bygget opp av tre samarbeidende komponenter:

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   🖥️  BRUKERGRENSESNITT                        │
│   Nettleserbasert skjema der brukeren limer inn tekst       │
│   og ser resultater                                         │
│                                                             │
│               │  sender tekst ▼                             │
│                                                             │
│   🔌  MELLOMTJENER (MCP Server)                            │
│   Koordinerer kommunikasjonen og gjør klassifiserings-      │
│   verktøyet tilgjengelig for andre AI-systemer              │
│                                                             │
│               │  videresender ▼                             │
│                                                             │
│   🧠  AI-MOTOR (API + Lokal LLM)                          │
│   Kjører selve klassifiseringsprosessen:                    │
│   • Leser og forstår patentteksten (AI-modell)             │
│   • Søker i CPC-databasen (260 000+ koder)                 │
│   • Scorer og validerer forslag                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Hvorfor tre deler?

| Komponent | Fordel |
|-----------|--------|
| **Brukergrensesnitt** | Kan endres uavhengig — f.eks. nytt design uten å røre AI-en |
| **Mellomtjener** | Gjør at andre verktøy (f.eks. Claude, Copilot) også kan bruke klassifiserings-motoren |
| **AI-motor** | Kan oppgraderes med bedre AI-modeller uten å påvirke resten |

---

## 6. Datakilder

Systemet bruker **offisielle CPC-data** — ikke egne tolkninger:

| Datakilde | Innhold | Opprinnelse |
|-----------|---------|-------------|
| **CPC XML-skjemafiler** | Alle 260 000+ CPC-koder med offisielle definisjoner | EPO (European Patent Office) |
| **Kunnskapsgrafen** | Hierarkisk graf med alle koder, foreldre-barn-relasjoner, og navigerbare forbindelser mellom CPC-klasser | Bygget automatisk fra XML-filene |
| **Semantiske embeddings (Sentence-BERT)** | AI-fingeravtrykk av alle 260 000+ CPC-titler, brukt til å søke etter *mening* i stedet for nøkkelord | Generert med embedding-modellen |
| **AI-språkmodell** | Forståelse av naturlig språk og tekniske begreper | Lokal AI — ingen data sendes ut |

> 🔒 **Personvern:** All behandling skjer **lokalt **. Ingen patenttekst sendes til eksterne skytjenester. Embeddings og kunnskapsgrafen er lagret lokalt .

---

## 7. Hva er nytt sammenlignet med tradisjonell metode?

| Aspekt | Tradisjonell metode | Vårt system |
|--------|-------------------|-------------|
| **Hastighet** | 15–30 min per søknad | Under 1 minutt |
| **Dekning** | Gransker husker kanskje 500–1000 koder | Søker i alle 260 000+ koder |
| **Konsistens** | Varierer mellom granskere | Samme tekst → samme resultat |
| **Dokumentasjon** | Gransker noterer kode uten begrunnelse | Systemet gir skriftlig begrunnelse for hver foreslått kode |
| **Oppdatering** | Krever opplæring ved endringer i CPC | Oppdater XML-filer → systemet tilpasser seg automatisk |
| **Kvalitetskontroll** | Manuell stikkprøve | Automatisk flerfase-validering innebygd |

---

## 8. Begrensninger — viktig å vite

| Begrensning | Forklaring |
|-------------|------------|
| 🤖 **AI er et verktøy, ikke en erstatning** | Systemet *foreslår* koder — det er fortsatt granskerens oppgave å godkjenne |
| 📝 **Kvaliteten avhenger av teksten** | Korte eller vage beskrivelser gir dårligere resultater |
| 🔄 **Krever oppdatering** | Når EPO oppdaterer CPC-skjemaet, må vi laste ned nye XML-filer |
| ⏳ **Førstegangsoppsett tar tid** | Bygging av kunnskapsgrafen tar 10–20 minutter (kun første gang) |

---

## 9. Mulige utvidelser

Her er noen retninger vi kan ta videre:

| Mulighet | Beskrivelse |
|----------|-------------|
| 📊 **Batch-klassifisering** | Klassifiser mange søknader samtidig (f.eks. import fra en mappe) |
| 🔗 **Integrasjon med eksisterende verktøy** | Koble systemet til eksisterende saksbehandlingssystem |
| 📈 **Lærende system** | La granskere gi tilbakemelding som forbedrer fremtidige forslag |
| 🌍 **Flerspråklig støtte** | Klassifisere patenter skrevet på andre språk enn engelsk |
| 📋 **Automatisk rapport** | Generere ferdig klassifiseringsrapport med begrunnelse |

---

## 10. Oppsummering

```
┌──────────────────────────────────────────────────┐
│                                                  │
│  📝  Bruker limer inn patenttekst               │
│                    ↓                             │
│  🧠  AI forstår innholdet og trekker ut          │
│      tekniske begreper                           │
│                    ↓                             │
│  🕸️  Kunnskapsgrafen søker semantisk            │
│      i 260 000+ CPC-koder (AI-embeddings)       │
│                    ↓                             │
│  🔍  Rangering basert på mening + nøkkelord     │
│                    ↓                             │
│  ✅  Validering og kvalitetskontroll             │
│                    ↓                             │
│  🏆  Rangerte forslag med begrunnelse            │
│                                                  │
│  ⏱️  Alt på under 1 minutt                      │
│  🔒  All data forblir lokalt                     │
│                                                  │
└──────────────────────────────────────────────────┘
```

 