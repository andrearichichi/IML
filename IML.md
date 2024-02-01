
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



 

