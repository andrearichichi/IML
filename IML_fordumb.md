#IML for DUMB

###**PERCEPTRON ALGORITHM**

Immagina che l'algoritmo Perceptron sia come un giudice che deve decidere se una frutta è una mela o una banana basandosi su alcune caratteristiche come il colore, la forma e la consistenza. 

**Features (Caratteristiche):**
Queste sono le informazioni che il giudice osserva. Per la nostra frutta, queste potrebbero essere:
- \( f_1 \): Rosso = 1, Non Rosso = 0
- \( f_2 \): Tonda = 1, Non Tonda = 0
- \( f_3 \): Morbida = 1, Dura = 0

**Pesi (Weights):**
Il giudice dà un'importanza diversa a ciascuna caratteristica, questa importanza è il "peso". Se il colore è molto importante per riconoscere una mela, allora avrà un peso maggiore. I pesi possono essere pensati come il fattore di quanto il giudice si fida di ciascuna caratteristica.

**Etichette (Labels):**
Queste sono le risposte corrette che il giudice cerca di indovinare. Per la frutta, potrebbe essere "Mela" o "Banana", che durante l'addestramento sono rappresentate da 1 o -1.

**L'Algoritmo:**
1. Il giudice inizia con un'idea casuale (pesi iniziali).
2. Guarda una frutta e prova a indovinare se è una mela o una banana basandosi sulle caratteristiche e sull'importanza (pesi) che ha dato a ciascuna.
3. Se indovina, fantastico! Non cambia nulla.
4. Se sbaglia, cambia l'importanza (pesi) delle caratteristiche in modo da ricordarsi di fare una scelta migliore la prossima volta che vede una frutta simile.
5. Ripete il processo guardando molte frutte, imparando dai suoi errori finché non diventa molto bravo a indovinare.

Il Perceptron continua questo processo finché non riesce a classificare correttamente tutte le frutte che ha visto, o per un numero stabilito di volte. Questo processo è il cuore dell'apprendimento del Perceptron, che lo aiuta a prendere decisioni accurate basandosi su ciò che ha imparato dalle sue esperienze passate.

---

###**OvA** (One-vs-All)
Immagina di avere un cesto di frutta e di voler insegnare a un computer a riconoscere se una frutta è una mela, un'arancia o una banana. L'approccio One-vs-All (OvA) è come se tu addestri tre giudici separati:

1. Il primo giudice impara a riconoscere le mele da tutto il resto (non mele).
2. Il secondo giudice impara a riconoscere le arance da tutto il resto (non arance).
3. Il terzo giudice impara a riconoscere le banane da tutto il resto (non banane).

Quando arriva una nuova frutta, chiedi a tutti e tre i giudici cosa pensano. Se la frutta è una mela, il primo giudice sarà il più sicuro della sua scelta, mentre gli altri due non saranno molto sicuri. Alla fine, scegli la decisione del giudice più sicuro e dici: "Questa frutta è una mela".

Questo è essenzialmente ciò che fa l'algoritmo OvA: crea un "giudice" per ogni tipo di frutta, poi per ogni nuova frutta sceglie il "giudice" che sembra più sicuro della sua decisione.

--- 

###**AvA** (All-vs-All)

Certamente! Immagina di dover classificare animali in tre categorie: gatti, cani e uccelli. L'approccio All-vs-All (AVA) funziona così:

1. **Decomposizione**: Invece di avere un solo classificatore per distinguere tra gatti, cani e uccelli, creiamo diversi classificatori per ogni possibile coppia di animali. Quindi, avremo tre classificatori binari: uno per distinguere gatti da cani, uno per gatti da uccelli e uno per cani da uccelli.

2. **Addestramento**: Ogni classificatore binario viene addestrato a riconoscere due animali specifici. Ad esempio, il classificatore per "gatti vs. cani" imparerà a distinguere tra gatti e cani, mentre il classificatore per "gatti vs. uccelli" imparerà a distinguere tra gatti e uccelli.

3. **Classificazione**: Quando riceviamo un nuovo animale da classificare, lo valutiamo con tutti e tre i classificatori. Ognuno di essi darà un voto, dicendo se l'animale somiglia più a un gatto o a un cane o a un uccello.

4. **Decisione finale**: Alla fine, contiamo i voti dati da ciascun classificatore. Se ad esempio "gatti vs. cani" e "gatti vs. uccelli" dicono che è più simile a un gatto, allora concludiamo che l'animale è un gatto.

In breve, l'AVA scompone il problema di classificare tra più classi in una serie di problemi più piccoli di classificazione binaria, dove ogni classificatore si occupa di confrontare due classi specifiche. Alla fine, combiniamo i voti di tutti i classificatori per determinare a quale classe appartiene l'oggetto da classificare.


---

##Loss Function

### 0/1 Loss
Immaginiamo di avere un semplice problema di classificazione: prevedere se farà sole o pioverà domani. Il tuo modello fa una predizione e, basandosi sulla realtà osservata il giorno successivo, la predizione viene valutata.

- **Esempio:** Previsione: Sole, Realtà: Sole → Perdita: **0** (corretto)
- **Esempio:** Previsione: Sole, Realtà: Pioggia → Perdita: **1** (errato)

Questa funzione di perdita è molto diretta ma non fornisce una via chiara per l'aggiustamento dei parametri del modello, dato che non c'è una gradazione dell'errore; è semplicemente giusto o sbagliato.

### Convex Loss Functions
Consideriamo l'Errore Quadratico Medio (MSE) per un problema di regressione, come prevedere la temperatura di domani. MSE fornisce una misura continua dell'errore che può essere minimizzata efficacemente.

- **Esempio:** Previsione: 20°C, Realtà: 22°C → Perdita: \( (20 - 22)^2 = 4 \)

Se il modello prevede 21°C invece che 20°C, la perdita sarebbe \( (21 - 22)^2 = 1 \), mostrando come piccoli aggiustamenti nella previsione influenzano la perdita in modo continuo e prevedibile, facilitando l'ottimizzazione.

### Surrogate Loss Functions
Nel contesto della classificazione binaria, dove dobbiamo decidere tra due classi (es. email è spam o no), la Hinge Loss può servire come funzione di perdita surrogata. Supponiamo che le etichette siano +1 per spam e -1 per non spam, e il modello emette un punteggio, dove valori più alti indicano maggiore confidenza che l'email sia spam.

- **Esempio:** Previsione (punteggio): +2 per un'email effettivamente spam (etichetta: +1) → Perdita bassa o nulla, a seconda del margine.
- **Esempio:** Previsione (punteggio): -1 per un'email effettivamente spam (etichetta: +1) → Perdita elevata.

La Hinge Loss penalizza le predizioni che sono sia errate sia quelle corrette ma con confidenza insufficiente, promuovendo margini di decisione più ampi tra le classi.

**Riepilogo con un'esemplificazione visiva:**

- **0/1 Loss:** È come un interruttore luce: acceso o spento (corretto o errato). Non fornisce indicazioni su come essere "meno sbagliato".
- **Convex Loss (MSE):** È come regolare l'intensità di una lampadina con un dimmer. Più ti avvicini alla luminosità desiderata, minore è l'errore.
- **Surrogate Loss (Hinge Loss):** È come cercare di centrare un bersaglio con una freccetta, dove avvicinarsi al centro riduce la perdita, ma c'è anche un margine entro cui vuoi che la freccetta cada per considerarla un successo.

Questi esempi dovrebbero aiutarti a comprendere meglio come funzionano queste diverse funzioni di perdita e il loro impatto sull'addestramento dei modelli di machine learning.

---

###Gradient Descent

Possiamo pensare al Gradient Descent come al processo di trovare il punto più basso in una valle. Sei in cima a una collina e vuoi arrivare in fondo nel punto più basso. Ogni passo che fai è determinato da dove la collina è più ripida; in pratica, cerchi di andare sempre nella direzione che ti porta più velocemente verso il basso.

La formula che hai visto nell'immagine è come la ricetta per decidere come fare ciascun passo. Ecco la spiegazione di ogni parte della formula:

- **\( w_j \)**: Questo è il punto in cui ti trovi sulla collina.
- **\( \eta \)**: Questo è il tasso di apprendimento, che ti dice quanto grandi dovrebbero essere i tuoi passi. Se è troppo grande, potresti scivolare troppo velocemente e superare il punto più basso. Se è troppo piccolo, ci metterai molto tempo a raggiungere il fondo.
- **\( \frac{\partial}{\partial w_j} \text{loss}(w) \)**: Questa parte ti dice quanto è ripido il pendio dove ti trovi, dandoti la direzione in cui dovresti muoverti per andare verso il basso. È come guardare intorno a te e scegliere il sentiero che scende più ripidamente.

Ora, per il segno negativo:

- **Perché negativo?**: Il segno meno è come una bussola che ti dice di andare nella direzione opposta a quella in cui la collina sale, quindi verso il basso. Senza il segno meno, cammineresti verso l'alto anziché scendere.

Mettendo tutto insieme, la formula completa:

\[ w_j = w_j - \eta \frac{\partial}{\partial w_j} \text{loss}(w) \]

...è come dire: "Il mio nuovo punto (dove mi trovo dopo aver fatto un passo) è il punto dove ero prima, meno un piccolo passo nella direzione che mi porterà verso il basso più rapidamente."

In pratica, ripeti questo processo di scelta della direzione e di fare un passo molte volte, e con ogni passo ti avvicinerai sempre più al punto più basso della valle, che è il tuo obiettivo.