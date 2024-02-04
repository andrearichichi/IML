
# Machine Learning
### **Supervised Learning**
L'algoritmo apprende da dati etichettati, cercando di fare previsioni accurate basate su nuovi dati.

1. **Classificazione**:
	- **Tipo di Output**: Output categorico o discreto.
	- **Esempi di Algoritmi**: Alberi decisionali, reti neurali, K-nearest neighbors, support vector machines.
	- **Applicazioni**: Riconoscimento delle impronte digitali, classificazione delle email, diagnosi medica.

2. **Regressione**:
	-**Tipo di Output**: Output continuo o quantitativo.
	-**Esempi di Algoritmi**: Regressione lineare, regressione polinomiale, regressione con alberi decisionali.
	-**Applicazioni**: Stima dei prezzi delle case, previsioni meteo, previsione delle vendite.

3. **Ranking**:
	- **Tipo di Output**: Ordinamento di elementi in base a rilevanza o importanza.
	- **Esempi di Algoritmi**: RankNet, LambdaRank, Learning to Rank.
	- **Applicazioni**: Motori di ricerca, sistemi di raccomandazione (suggerimenti di prodotti o contenuti).

---
### **Unsupervised Learning**
L'algoritmo cerca di identificare pattern o strutture nei dati senza etichette.

1. **Clustering**:
   - **Output**: Identifica gruppi naturali all'interno dei dati.
   - **Algoritmi**: K-means, clustering gerarchico, DBSCAN.
   - **Applicazioni**: Segmentazione delle immagini, classificazione biologica, sistemi di raccomandazione.

2. **Anomaly Detection**:
   - **Output**: Rileva dati che differiscono significativamente dalla norma.
   - **Algoritmi**: Isolation Forest, Local Outlier Factor, Autoencoder.
   - **Applicazioni**: Rilevamento di frodi, manutenzione predittiva, sicurezza informatica.

3. **Dimensionality Reduction**:
   - **Output**: Riduce il numero di variabili mantenendo le informazioni principali.
   - **Algoritmi**: PCA (Analisi dei Componenti Principali), t-SNE, LDA.
   - **Applicazioni**: Semplificazione dei dati, visualizzazione, maggior efficienza del training di modelli.


---
### **Reinforcement Learning**
Il Reinforcement Learning (apprendimento per rinforzo) è un tipo di apprendimento automatico dove un agente impara a prendere decisioni ottimizzando il suo comportamento attraverso l'interazione con un ambiente. Ecco una panoramica sintetica:

- **Obiettivo**: Massimizzare la ricompensa totale attraverso le azioni.
- **Componenti Principali**:
  - **Agente**: Entità che prende decisioni.
  - **Ambiente**: Contesto in cui l'agente opera.
  - **Stati**: Rappresentazione della situazione dell'agente.
  - **Azioni**: Scelte che l'agente può fare.
  - **Ricompense**: Feedback dall'ambiente in risposta alle azioni.
- **Approccio**: L'agente prova diverse azioni e impara dalle ricompense/punizioni ricevute.
- **Algoritmi**: Q-learning, Deep Q Network (DQN), Policy Gradients, A3C.
- **Applicazioni**: Giochi (es. scacchi, Go), robotica, sistemi di navigazione autonomi, ottimizzazione di processi industriali.

---
### Other learning variations
How are we getting the data: 
- online vs. offline learning. 

Type of models:

-   generative vs. discriminative
-    parametric vs. non-parametric


---

### Training & Test Set

-   **Training Set**:
    -   **Scopo**: Utilizzato per addestrare il modello di machine learning.
    -   **Uso**: Il modello impara a riconoscere pattern o a fare previsioni basate su questi dati.
    -   **Dimensione**: Tipicamente, una porzione più grande del dataset (ad esempio, 70-80%).
    
-   **Test Set**:
    -   **Scopo**: Utilizzato per valutare le prestazioni del modello dopo l'addestramento.
    -   **Uso**: Testa quanto bene il modello funziona con dati nuovi e non visti durante l'addestramento.
    -   **Dimensione**: Generalmente una porzione più piccola del dataset (ad esempio, 20-30%).


## ML Ingredients:

- **TASK:** La funzione che assegna ad ogni input X un output Y. La natura di X e Y dipendono dal Task.

- **DATA:** Sono informazioni rappresentate come una distribuzione di probabilità per risolvere problemi specifici (P_data: Simboleggia la distribuzione di probabilità dei dati nel machine learning).  
	-   Per **Classificazione e Regressione**:
    
	    -   **P_data ∈ Δ(ℵ × 𝒴)**: La distribuzione dei dati è definita su due spazi: uno per gli input (ℵ) e uno per gli output (𝒴). Lo scopo è prevedere gli output dagli input.
	-   Per **Anomaly Detection, Clustering e Riduzione della Dimensionalità**:
    
	    -   **P_data ∈ Δ(ℵ)**: La distribuzione dei dati è definita solo sull'input (ℵ). L'obiettivo è comprendere la struttura e la distribuzione degli input senza prevedere un output.

- **MODEL and HYPOTHESIS SPACE**: Un modello è come un "programma" per risolvere il problema.
Un set di modelli definiscono un **Hypothesis space**, all'interno del quale l'algoritmo di apprendimento cerca una soluzione.

- **OBJECTIVE**:
	- **Obiettivo - Il Target Ideale**: Trovare la funzione che minimizza l'errore di generalizzazione su tutti i possibili dati. Questo è un obiettivo teorico, poiché lo spazio delle possibili funzioni è troppo ampio per una ricerca effettiva.

	- **Obiettivo - Il Target Fattibile**: Dato che non possiamo esplorare tutto lo spazio delle funzioni, restringiamo la ricerca a un sottoinsieme praticabile e cerchiamo la funzione  che minimizza l'errore all'interno di questo sottoinsieme. Questo è ancora un obiettivo teorico perché non conosciamo la distribuzione esatta dei dati.

	- **Obiettivo - Il Target Effettivo**: Utilizziamo un set di dati di allenamento concreto e cerchiamo la funzione  che minimizza l'errore su questo set. Questo è l'obiettivo pratico del machine learning, che porta a un modello che può essere effettivamente addestrato e valutato.

	La **funzione di Errore** in machine learning è un modo per misurare quanto bene un modello predittivo sta facendo il suo lavoro, cioè quanto bene sta predicedo i dati. In particolare:
	- **Funzione di Errore:**
		- Misura la performance del modello predittivo.
		- Valuta la precisione delle previsioni rispetto ai valori reali.

	- **Pointwise Loss:**
		- Calcola l'errore per un singolo punto dati.
		- Indica quanto la previsione del modello differisce dal valore atteso per quel punto.

	- **Calcolo dell'Errore Complessivo:**
		- Somma il pointwise loss per tutti i punti del set di dati.
		- Dividi la somma totale per il numero di punti per ottenere l'errore medio.
		- Fornisce una valutazione generale dell'errore del modello sul set di dati completo.
	

- **LEARNING ALGORITHM:** Hanno il compito di trovare la funzione che minimizza l'errore di previsione sui dati. La soluzione ideale sarebbe trovare il minimo globale, ma spesso l'algoritmo può fermarsi in un minimo locale, un punto in cui l'errore è minimo rispetto all'ambiente immediato ma non è il minimo assoluto.


---

OVERFITTING & UNDERFITTING


| Aspetto         | Overfitting                             | Underfitting                            |
|-----------------|-----------------------------------------|-----------------------------------------|
| **Definizione** | Modello troppo aderente ai dati di addestramento. | Modello troppo semplice, non apprende abbastanza. |
| **Cause**       | Complessità eccessiva, addestramento eccessivo. | Complessità insufficiente, addestramento insufficente. |
| **Conseguenze** | Alta accuratezza su addestramento, bassa su test. | Bassa accuratezza su entrambi addestramento e test. |
| **Soluzioni**   | Semplificare modello, più dati, regolarizzazione. | Aumentare complessità, più caratteristiche, più addestramento. |


---



### Generalizzazione

- **Definizione**: La capacità di un modello di machine learning di performare bene su nuovi, non visti dati, che non sono stati utilizzati durante il processo di addestramento. La generalizzazione è l'obiettivo finale dell'addestramento di un modello, indicando quanto bene il modello possa estendere ciò che ha imparato ai dati generali al di fuori del suo set di addestramento.
- **Importanza**: Un modello che generalizza bene è utile nella pratica, poiché può fare previsioni accurate su dati non visti.
- **Misurazione**: La generalizzazione di un modello è tipicamente valutata usando un set di dati di test, separato e non utilizzato durante l'addestramento, per simulare come il modello si comporterebbe su nuovi dati.


### Miglioramento della Generalizzazione:
- **Ridurre la complessità**: Scegliere modelli con meno parametri o con strutture più semplici.
- **REGOLARIZZAZIONE***: Aggiungere un termine di penalità alla funzione di perdita per limitare l'ampiezza dei parametri.
- **Aumentare i dati**: Più dati di addestramento possono migliorare la capacità del modello di generalizzare.
- **Data Augmentation**: Modificare i dati esistenti per creare nuovi esempi di addestramento.
- **Fermare presto**: Interrompere l'addestramento prima che il modello inizi a sovradattarsi.
- **Iniettare rumore**: Aggiungere rumore durante l'addestramento per rendere il modello meno sensibile alle fluttuazioni dei dati di addestramento.

>	La **REGOLARIZZAZIONE** è uno strumento per migliorare la generalizzazione di un modello. Riducendo l'overfitting, un modello regolarizzato tende ad avere una migliore performance su dati non visti, poiché è meno probabile che memorizzi i dettagli specifici o il rumore presente nel set di addestramento. In sostanza, la regolarizzazione aiuta a creare modelli che sono robusti e affidabili, piuttosto che modelli che funzionano eccezionalmente bene sui dati di addestramento ma falliscono nel predire accuratamente nuovi dati.


---

###K-Nearest Neighbor
L'algoritmo K-Nearest Neighbors (KNN) è un metodo semplice e intuitivo usato nel machine learning per la classificazione e la regressione. Funziona seguendo questi passaggi:

1. **Scegliere il valore di K**: Si decide quanti vicini (K) considerare per fare la predizione. Un \(K\) piccolo può essere sensibile al rumore, mentre un \(K\) grande può rendere il modello troppo generico.

2. **Calcolare la distanza**: Si misura quanto ogni punto nel set di addestramento è distante dal punto da classificare, usando metriche come la distanza euclidea.

3. **Trovare i \(K\) vicini più prossimi**: Si ordinano i punti per distanza crescente e si selezionano i primi \(K\).

4. **Votazione o media per la predizione**:
   - **Classificazione**: Si assegna al punto la classe più frequente tra i suoi \(K\) vicini.
   - **Regressione**: Si calcola la media dei valori dei \(K\) vicini.

**Vantaggi** includono la semplicità di implementazione e l'efficacia per set di dati piccoli. 
**Svantaggi** sono la lentezza su set di dati grandi e la sensibilità alla scelta di \(K\) e alla distribuzione dei dati.

E' un algoritmo con **complessità lineare O(n)** 

---


###**BIAS**

Il BIAS è La forza delle assunzioni che un modello fa sui dati.

1. Classificatori a **Basso Bias**:
   - Fanno poche assunzioni sui dati.
   - Sono flessibili e si adattano bene a varie distribuzioni dei dati.
   - Esempi:
     - k-Nearest Neighbors (k-NN)
     - Decision Trees (DT)

<br>

2. Classificatori ad **Alto Bias**:
   - Fanno assunzioni forti sui dati.
   - Meno flessibili, possono non adattarsi bene a dati complessi/non lineari.
   - Esempio di assunzione ad alto bias: Separabilità Lineare
     - Funziona bene se i dati sono linearmente separabili.
     - Utilizza linee rette in 2D o iperpiani in dimensioni superiori.

---
###**Modelli Lineari**

Un modello lineare è un modello che assume che i dati possano esserere **separabili linearmente**.

<img src="linear.png" width="50%">
<br>



**Separazione Lineare**: 
   - 2D, linea
   - 3D, piano
   - ND, hyper-plane

---

**Online Learning**

L'Online Learning Algorithm è un approccio al machine learning in cui il modello viene addestrato sequenzialmente, una singola osservazione (o un piccolo gruppo di osservazioni) per volta. Ecco una sintesi:

- **Input Sequenziali**: Il modello riceve i dati in sequenza, non tutti in una volta.
- **Aggiornamenti Continui**: Il modello si aggiorna ad ogni nuova osservazione.
- **Adatto a Dati in Flusso**: Ideale per sistemi con flussi di dati continui.
- **Risparmio Memoria**: Non richiede di memorizzare l'intero dataset.
- **Adattabilità**: Il modello può adattarsi ai cambiamenti nei pattern dei dati.
- **Esempi di Algoritmi**: Perceptron Online, Stochastic Gradient Descent.

L'online learning è particolarmente utile in contesti dove i dati cambiano nel tempo o sono troppo grandi per essere processati in batch.

---
**BATCH vs ONLINE**



| Caratteristica | Online Learning | Batch Learning |
|----------------|-----------------|----------------|
| Dati           | Sequenziali     | Completi       |
| Memoria        | Basso consumo   | Alto consumo   |
| Adattabilità   | Elevata         | Bassa          |
| Applicazioni   | Tempo reale, flussi continui | Dataset statici e disponibili |


---

###**PERCEPTRON Algorithm**


1. **Inizializzazione**: Imposta i pesi \( w_i \) e il bias \( b \) a valori iniziali (spesso zero).

2. **Ciclo di Addestramento**:
   - Ripeti fino alla convergenza o per un numero prefissato di iterazioni.
   - Per ogni esempio di addestramento \( (f_1, f_2, ..., f_n, \text{etichetta}) \):
     - Calcola l'output del modello corrente per l'esempio di addestramento.
     - Se la classificazione è corretta, non fare nulla.
     - Se la classificazione è errata:
       - Aggiorna ogni peso \( w_i \): \( w_i = w_i + f_i \times \text{etichetta} \)
       - Aggiorna il bias \( b \): \( b = b + \text{etichetta} \)

3. **Convergenza**:
   - Il processo si ripete fino a quando non ci sono più errori di classificazione, o fino al raggiungimento del numero massimo di iterazioni stabilito.

<img src="perceptron.png" width="70%">
<br>

---

###**BINARY CLASSIFICATION**

- **Etichette**: Sì o No, Positivo o Negativo, 1 o 0.
- **Obiettivo**: Assegnare ogni elemento a una delle due categorie.
- **Algoritmi**: K-NN e PERCEPTRON

---

###**MULTI-CLASSIFICATION**

- **Etichette**: Categoria A, Categoria B, Categoria C, ... (oltre due categorie possibili).
- **Obiettivo**: Assegnare ogni elemento a una delle molteplici categorie disponibili.
- **Algoritmi**: OVA, AVA, 

---
### **OvA** (one-vs-all)

L'approccio One-vs-All (OvA) è una strategia usata per estendere algoritmi di classificazione binaria, come il Perceptron, a problemi di classificazione multiclasse. Invece di costruire un singolo classificatore che deve distinguere tra più di due classi, OvA costruisce un classificatore separato per ogni classe.


1. **Costruzione dei Classificatori**:
   - Per ogni classe \( C_i \), viene addestrato un classificatore \( f_i \) che distingue gli esempi di quella classe dal resto degli esempi (tutte le altre classi).

2. **Addestramento**:
   - Durante l'addestramento del classificatore \( f_i \), gli esempi della classe \( C_i \) sono etichettati positivamente, mentre tutti gli altri esempi sono etichettati negativamente.

3. **Classificazione**:
   - Per classificare un nuovo esempio, tutti i classificatori \( f_i \) vengono eseguiti.
   - L'etichetta finale assegnata all'esempio è quella del classificatore che ha la più alta fiducia o score di output nel riconoscere l'esempio.

**Vantaggi**:
   - Semplice da implementare.
   - Scalabile per un numero elevato di classi.

**Svantaggi**:
   - Potrebbe non essere efficiente se il numero delle classi è molto grande perché richiede di addestrare un classificatore per ogni classe.
   - I classificatori possono essere sbilanciati se le classi sono molto disuguagliate in termini di numero di esempi.

OvA è particolarmente utile perché consente di sfruttare algoritmi che sono nati per la classificazione binaria anche in contesti dove ci sono molte classi da distinguere.

---

###AVA (All-vs-All)

L'approccio All-vs-All (AVA) nel machine learning è utilizzato per problemi di classificazione multiclasse. Invece di addestrare un unico classificatore per distinguere tutte le classi, AVA suddivide il problema in numerosi problemi di classificazione binaria. Ecco i passaggi principali:

1. **Decomposizione**: Per un insieme di K classi, vengono addestrati \( K \times (K - 1) / 2 \) classificatori binari unici, uno per ogni possibile coppia di classi. Ad esempio, con 4 classi ci sono 6 classificatori binari distinti.

2. **Addestramento**: Ogni classificatore binario viene addestrato per distinguere tra due classi specifiche. Un'etichetta positiva è assegnata a una classe, mentre l'etichetta negativa all'altra.

3. **Classificazione**: Quando si deve classificare un nuovo dato, viene valutato da tutti i classificatori binari. Ogni classificatore fornisce un voto alla classe "positiva" e un voto alla classe "negativa" in base alla predizione.

4. **Decisione finale**: Dopo che tutti i classificatori hanno votato, si conteggiano i voti per ciascuna classe. La classe con il maggior numero di voti è considerata la classe finale assegnata al dato.

L'approccio AVA è utile per problemi in cui le classi sono difficili da separare linearmente, e può portare a una maggiore precisione rispetto ad altri metodi multiclasse come il One-vs-All. Tuttavia, richiede un grande numero di classificatori, rendendolo computazionalmente costoso con un aumento delle classi.

---
###MACROAVERAGING vs MICROAVERAGING

**MACROaveraging**
- **Calcola**: Media delle metriche per classe.
- **Peso Classi**: Uguale per ogni classe.
- **Uso**: Importanza eguale a tutte le classi.

>**Esempio Macroaveraging**
>Supponiamo di avere un sistema di classificazione con 3 classi (A, B, C), e abbiamo calcolato la precisione per ogni classe come segue:
> - Precisione Classe A: 0.5
> - Precisione Classe B: 0.8
> - Precisione Classe C: 0.7

> **Precisione Macroaveraged**:
>\[ \frac{0.5 + 0.8 + 0.7}{3} = \frac{2.0}{3} = 0.67 \]

**MICROaveraging**
- **Calcola**: Metriche globali aggregando tutti i campioni.
- **Peso Classi**: Basato sul volume dei campioni.
- **Uso**: Riflette prestazioni su volumi maggiori di campioni.



>**Esempio Microaveraging**
>Utilizzando lo stesso esempio ma calcolando la precisione microaveraged:
>- Supponiamo che Classe A abbia 10 TP (veri positivi) e 10 FP (falsi positivi).
>- Classe B ha 40 TP e 10 FP.
>- Classe C ha 30 TP e 20 FP.

>**Precisione Microaveraged**:
\[ \frac{10 + 40 + 30}{10 + 10 + 40 + 10 + 30 + 20} = \frac{80}{120} = 0.67 \]

---

### Loss Function 

Nel machine learning, le funzioni di perdita valutano l'accuratezza delle previsioni di un modello, con l'obbiettivo di minimizzare gli errori.

**Tipi principali:**

- **0/1 Loss**: Misura se una previsione è corretta (0) o sbagliata (1). Semplice ma non differenziabile, quindi meno usata per l'ottimizzazione basata su gradienti.
- **Convex Loss**: Funzioni dove esiste un unico minimo globale, facilitando l'ottimizzazione. Esempi: MSE, cross-entropy.
- **Surrogate Loss**: Sostituiscono funzioni non ottimizzabili (come 0/1 loss) con alternative differenziabili, adatte per ottimizzazione. Esempi: log loss, hinge loss.

---


###Gradient Descent

Il Gradient Descent è un algoritmo di ottimizzazione fondamentale nel campo del machine learning per minimizzare la funzione di perdita di un modello.

In particolare, questo è il funzionamento:

- Partendo da un punto w.
- Ripeto:
   -   scelgo una dimensione.
   -   mi muovo in quella dimensione per abbassare l'errore (tramite derivate).
   \[ w_j = w_j - \eta \frac{\partial}{\partial w_j} \text{loss}(w) \]
   
<img src="gradientd.png" width="70%">

**Learning Rate**: iperparametro fondamentale che determina la grandezza dei "passi".

---


### Regolarizzazione

1. **Definizione:** La regolarizzazione è un termine aggiuntivo nella funzione di perdita che penalizza i valori estremi dei parametri (pesi) del modello.
   
2. **Obiettivo:** Impedisce al modello di adattarsi troppo ai dati di addestramento, incoraggiando i pesi a rimanere piccoli e, quindi, il modello a essere più semplice.

3. **Bias:** Aggiunge un pregiudizio nella scelta dei pesi, spingendo il modello a preferire pesi più piccoli, o zero per caratteristiche meno importanti.

 **Tipi Comuni di Regolarizzatori**

1. **L1 Regularization (Sum of Weights):** \( r(w,b) = \sum |w_j| \)
   - Penalizza la somma assoluta dei pesi.
   - Può portare alcuni pesi a essere esattamente zero, offrendo una selezione di caratteristiche "sparse".

2. **L2 Regularization (Sum of Squared Weights):** \( r(w,b) = \sum w_j^2 \)
   - Penalizza la somma dei quadrati dei pesi.
   - Tende a ridurre tutti i pesi, ma non necessariamente a zero, risultando in un modello con pesi uniformemente piccoli.

 **Differenze Chiave tra L1 e L2**

1. **L1 Regularization:**
   - Produce modelli più semplici e sparsi.
   - Utile quando alcune caratteristiche non sono rilevanti.

2. **L2 Regularization:**
   - Penalizza i pesi grandi più severamente (a causa del quadrato).
   - Mantiene tutti i pesi piccoli, ma non li annulla.
