# Predikce kvality vína pomocí hlubokého učení

Semestrální projekt do předmětu Deep Learning.

## 1. Cíl projektu

Cílem projektu je předpovídat kvalitu vína na základě jeho fyzikálně-chemických vlastností a porovnat vlastní neuronovou síť s moderním předtrénovaným modelem.

Projekt zahrnuje průzkum dat, předzpracování, návrh a trénování modelů, hledání hyperparametrů, experimentální vyhodnocení a analýzu chyb.

## 2. Dataset

Použit je veřejně dostupný dataset **Wine Quality** z UCI Machine Learning Repository.

Zdroj: https://archive.ics.uci.edu/dataset/186/wine+quality

Dataset obsahuje informace o červených a bílých portugalských vínech.

- Původní počet záznamů: 6 497
- Počet záznamů po odstranění duplicit: 5 320
- Počet vstupních atributů: 12
- Cílová proměnná: `quality`

Vstupní atributy představují 11 fyzikálně-chemických vlastností a informaci o typu vína (`wine_type`).

Data jsou rozdělena na trénovací (70 %), validační (15 %) a testovací (15 %) množinu. Standardizace číselných atributů je provedena pomocí `StandardScaler`, který se učí pouze na trénovacích datech.

## 3. Použité modely

### Baseline

`DummyRegressor` předpovídající průměrnou kvalitu vína z trénovací množiny.

### KNN

Klasický model `KNeighborsRegressor`, který slouží jako další referenční metoda.

### Model A – vlastní neuronová síť MLP

Neuronová síť vytvořená a trénovaná pomocí PyTorch od náhodné inicializace.

Finální konfigurace:

- Vstupní vrstva: 12 neuronů
- Skryté vrstvy: 128 a 64 neuronů
- Aktivační funkce: ReLU
- Výstupní vrstva: 1 neuron
- Optimalizátor: Adam
- Loss funkce: MSE
- Learning rate: 0,01
- Early stopping: patience 50
- Počet parametrů: 9 985

Pro zvýšení stability je vytvořen také ensemble tří nezávisle natrénovaných MLP.

### Model B – TabPFN-2.5

Moderní předtrénovaný model určený pro tabulková data.

Použita je regresní varianta `TabPFNRegressor`. Model je aplikován bez dodatečného gradientového fine-tuningu vah.

## 4. Experimenty

Projekt obsahuje:

1. Porovnání různých architektur MLP.
2. Hledání vhodného learning rate.
3. Experiment s různým množstvím trénovacích dat.
4. Ensemble několika neuronových sítí.
5. Opakování klíčových porovnání s různými náhodnými semínky.
6. Analýzu chyb podle skutečné kvality vína.
7. Porovnání výpočetní náročnosti a rychlosti inference.
8. Doplňkovou úlohu porovnávání kvality dvojic vín.

Pro opakované experimenty jsou použita semínka `42`, `123` a `2026`.

## 5. Výsledky

Závěrečné vyhodnocení na testovací množině:

| Model | MAE | RMSE |
|---|---:|---:|
| Baseline | 0,6926 | 0,8599 |
| KNN | 0,5403 | 0,6858 |
| MLP ensemble | 0,5247 | 0,6599 |
| TabPFN-2.5 | **0,5198** | **0,6476** |

Nižší hodnoty MAE a RMSE znamenají přesnější predikci.

TabPFN-2.5 dosáhl nejnižších chyb na testovacích datech. Rozdíly mezi modely je však nutné interpretovat s ohledem na velikost datasetu, variabilitu mezi běhy a výpočetní náročnost.

## 6. Použité technologie

- Python
- Google Colab
- PyTorch
- TabPFN
- NumPy
- pandas
- scikit-learn
- Matplotlib
- Seaborn
- joblib

## 7. Spuštění projektu

Projekt je vytvořen v jazyce Python v prostředí Google Colab. Pro jeho spuštění není nutné instalovat Python do počítače.

### 7.1 Otevření projektu

1. Stáhněte soubor `Wine_Quality_Deep_Learning_final.ipynb` z tohoto GitHub repozitáře.
2. Otevřete https://colab.research.google.com/.
3. Zvolte **File → Upload notebook** a nahrajte stažený soubor.
4. Doporučuje se nastavit GPU prostřednictvím **Runtime → Change runtime type → T4 GPU**, pokud je dostupná.
5. Spusťte notebook postupně od první buňky nebo pomocí **Runtime → Run all**.

Notebook obsahuje instalační příkazy pro potřebné knihovny. Dataset Wine Quality se načítá automaticky z veřejného zdroje UCI Machine Learning Repository.

Seznam potřebných knihoven je uveden v souboru `requirements.txt`. Pokud je tento soubor dostupný v pracovním prostředí, lze knihovny nainstalovat pomocí příkazu:

```bash
pip install -r requirements.txt
```

Při použití Google Colabu je možné využít instalační příkazy obsažené přímo v notebooku.

### 7.2 Získání přístupového klíče TabPFN-2.5 (Prior Labs)

Projekt využívá předtrénovaný model TabPFN-2.5 od společnosti Prior Labs. Notebook obsahuje buňku pro zadání přístupového klíče prostřednictvím proměnné `TABPFN_TOKEN`.

Postup získání klíče:

1. Otevřete stránku https://ux.priorlabs.ai/.
2. Zaregistrujte se nebo se přihlaste ke svému účtu.
3. V uživatelském účtu zkontrolujte a případně přijměte licenční podmínky modelu TabPFN-2.5.
4. V části určené pro správu API klíčů nebo přístupových tokenů vytvořte nový klíč, pokud je tato možnost pro váš účet dostupná.
5. Vygenerovaný klíč si ihned bezpečně zkopírujte a uložte.

Notebook obsahuje krok pro zadání přístupového tokenu TABPFN_TOKEN. Požadavky na ověření přístupu se mohou lišit podle verze knihovny TabPFN a způsobu získání modelových vah. V případě potřeby postupujte podle aktuálních pokynů společnosti Prior Labs.

**Důležité:** Celá hodnota nově vytvořeného klíče se může zobrazit pouze jednou, bezprostředně při jeho vytvoření. Po zavření okna už nemusí být možné ji znovu zobrazit. Pokud klíč ztratíte, bude potřeba vytvořit nový.

### 7.3 Vložení klíče do Google Colabu

Při spuštění notebooku se zobrazí výzva:

```text
Vloz svuj TabPFN API klic:
```

Do vstupního pole vložte vlastní klíč a potvrďte klávesou Enter.

Notebook používá následující kód:

```python
import os
from getpass import getpass

os.environ["TABPFN_TOKEN"] = getpass(
    "Vloz svuj TabPFN API klic: "
)
```

Funkce `getpass()` při zadávání klíč skryje, takže se jeho hodnota nezobrazuje v notebooku.

Klíč je uložen pouze do proměnné prostředí aktuální relace Google Colabu. Při restartování relace může být nutné zadat jej znovu.

**Bezpečnost:** Nikdy nevkládejte svůj skutečný API klíč přímo do zdrojového kódu ani jej nezveřejňujte na GitHubu. Každý uživatel používá vlastní přístupové údaje.

### 7.4 Stažení modelu a dokončení výpočtů

Po nastavení přístupu notebook pokračuje načtením modelu `TabPFNRegressor`. Při prvním použití může být nutné stáhnout předtrénované váhy modelu a ověřit oprávnění k jejich použití.

Projekt používá lokální knihovnu `tabpfn`, nikoli cloudového klienta `tabpfn-client`. Samotná instalace knihovny a oprávnění k použití modelových vah jsou dvě odlišné věci. Konkrétní požadavky na přístup se mohou lišit podle verze knihovny a podmínek Prior Labs.

Po úspěšném spuštění notebook vytvoří srovnání modelů, statistické tabulky, grafy, analýzu chyb a uloží checkpointy vlastních neuronových sítí.


## 8. Reprodukovatelnost

Projekt využívá pevně definované rozdělení dat na trénovací (70 %), validační (15 %) a testovací (15 %) množinu. Pro opakované experimenty jsou použita náhodná semínka `42`, `123` a `2026`.

Veškerý zdrojový kód, výsledky experimentů, grafy průběhu trénování, porovnání modelů a analýza chyb jsou součástí Jupyter notebooku.

Při spuštění notebooku se vytvářejí výstupní soubory a checkpointy modelů. Natrénované modely MLP, použitý scaler a konfigurace se ukládají do složky `ulozene_modely`.

Výstupní soubory vznikají v prostředí Google Colab a nejsou automaticky ukládány do GitHub repozitáře. Pro jejich uchování je nutné je samostatně stáhnout.

Potřebné knihovny jsou uvedeny v souboru `requirements.txt`. Výsledky experimentů se mohou mírně lišit podle verzí knihoven, použitého hardwaru a prostředí Google Colab.

## 9. Výpočetní prostředí

Experimenty byly provedeny v prostředí Google Colab s využitím CPU a GPU NVIDIA Tesla T4.

Při porovnávání časů je třeba zohlednit, že MLP je trénována od náhodné inicializace, zatímco TabPFN využívá předtrénované váhy.

## 10. Omezení projektu

- Dataset obsahuje nerovnoměrně zastoupené hodnoty kvality vína.
- Extrémní hodnoty kvality jsou zastoupeny malým počtem vzorků.
- Porovnání je provedeno na jednom rozdělení dat.
- Tři opakování poskytují pouze orientační odhad variability.
- Modely se liší výpočetní náročností i způsobem trénování.
- Výsledky nelze automaticky zobecnit na jiné datasety.

## 11. Využití generativní umělé inteligence

Při tvorbě projektu byly využity nástroje generativní umělé inteligence jako podpůrný prostředek při návrhu a úpravách zdrojového kódu, vysvětlování použitých metod, řešení technických problémů a přípravě dokumentace.

Experimenty byly provedeny v prostředí Google Colab. Výsledky a závěry vycházejí z provedených výpočtů a vyhodnocení použitých modelů.
  

## 12. Závěr

Projekt ukazuje možnosti použití hlubokého učení pro regresní úlohu nad tabulkovými daty.

Vlastní MLP dosahuje konkurenceschopných výsledků a umožňuje podrobně analyzovat vliv architektury a trénovacích hyperparametrů.

Předtrénovaný TabPFN dosáhl nejnižší chyby, avšak jeho použití je spojeno s odlišnými výpočetními nároky.

Projekt proto porovnává nejen kvalitu predikcí, ale také stabilitu, efektivitu a omezení obou přístupů.
