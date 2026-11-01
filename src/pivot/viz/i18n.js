// Languages of the viewer: its texts, and the names of objects and rooms.
// The page picks the browser's language (navigator.languages), which a menu
// can change; the choice is remembered on this browser.
// Inlined in the page before its module, which reads window.PivotI18n.
(() => {

const LANGUAGES = {
  en: { name: "English", speech: "en-US" },
  fr: { name: "Français", speech: "fr-FR" },
  de: { name: "Deutsch", speech: "de-DE" },
  es: { name: "Español", speech: "es-ES" },
  it: { name: "Italiano", speech: "it-IT" },
  zh: { name: "中文", speech: "zh-CN" },
};

const TEXTS = {
  en: {
    remove: "Remove", removed: "{name} taken out of the room. Undo puts it back.", removedWith: "Also taken out: {names}.",
    tourNext: "Next",
    tourSkip: "Skip",
    tourDone: "Let's go!",
    tourHelp: "Show the tour again",
    tour1: "Welcome to PIVOT! This is your room, rebuilt in 3D from a video. Drag to turn around it, scroll to zoom.",
    tour2: "On the left: choose the room, what to show (points, objects, floor) and the language.",
    tour3: "To move something, click it, then click where it should go. PIVOT checks the real geometry before anything moves: Check tests a move, Move does it, Undo goes back.",
    tour4: "Or simply ask, typed or spoken: \"Can the sofa go into the bedroom?\" Every answer is checked, and the moving plan gives the steps to print.",
    chooseRoom: "Choose a room", language: "Language",
    pointCloud: "Point cloud", objectBoxes: "Object boxes", objectVolumes: "Object volumes",
    walkableFloor: "Walkable floor", cutAbove: "Cut above",
    moveTitle: "Move an object",
    moveHint: "Click an object, then click where it should go: under a table, on a desk (light " +
      "objects are carried), anywhere. After a move, the next click picks a new object; to " +
      "change it before, use \"Change\", Ctrl-click or Esc. Every move is planned and checked " +
      "against the real geometry before it happens.",
    object: "Object", target: "Target", rotation: "Rotation", degrees: "degrees", keep: "keep",
    none: "none", change: "Change", check: "Check", move: "Move", undo: "Undo", reset: "Reset",
    planButton: "Moving plan (PDF)",
    planTitle: "The moves made so far, as numbered steps and a plan, to save as a PDF",
    makeRoom: "Make room & move",
    askTitle: "Ask PIVOT",
    askHint: "For example: \"Can chair 5 go under table 72?\", \"Put the trash can next to door " +
      "26 without blocking it.\" Every answer is checked against the geometry.",
    askPlaceholder: "Your request…", send: "Send", speak: "Speak", stop: "Stop",
    transcribing: "Transcribing…", micTitle: "Click, speak, click again",
    readAloud: "Read answers aloud",
    meta: "{objects} objects · {points} points",
    planning: "Planning…", error: "Error", moved: "Moved", feasible: "Feasible",
    pathLine: "path {length} m, rotation {rotation}°",
    effortLine: "effort {effort} m, tightest clearance {clearance}", overOneMeter: "> 1 m",
    notPossible: "Not possible",
    status_no_path: "no path", status_invalid_goal: "the spot is not free",
    status_invalid_start: "the object is stuck",
    status_search_limit_reached: "search limit reached",
    note: "Note: it {warning}.",
    inFront: "Note: it stands in front of the {label} (#{id}).",
    adjusted: "Target adjusted by {cm} cm to leave room around the object.",
    lookingForRoom: "Looking for objects to move out of the way…",
    undone: "Last change undone.", sceneReset: "Scene reset.",
    thinking: "Thinking…", checkedWith: "Checked with: ",
    budgetSpent: "The public demo has used its budget: run PIVOT yourself (see the README).",
    requestsLeft: "Public demo: {n} requests left today.",
    requestLeft: "Public demo: 1 request left today.",
    heardNothing: "I did not hear anything.", micUnavailable: "Microphone unavailable: ",
  },
  fr: {
    remove: "Supprimer", removed: "{name} retiré de la pièce. « Annuler » le remet.", removedWith: "Retiré aussi : {names}.",
    tourNext: "Suivant",
    tourSkip: "Passer",
    tourDone: "C'est parti !",
    tourHelp: "Revoir la visite guidée",
    tour1: "Bienvenue dans PIVOT ! Voici votre pièce, reconstruite en 3D à partir d'une vidéo. Faites glisser pour tourner autour, molette pour zoomer.",
    tour2: "À gauche : choisissez la pièce, ce qui est affiché (points, objets, sol) et la langue.",
    tour3: "Pour déplacer un objet, cliquez dessus, puis là où il doit aller. PIVOT vérifie la géométrie réelle avant tout déplacement : « Vérifier » teste, « Déplacer » exécute, « Annuler » revient en arrière.",
    tour4: "Ou demandez simplement, par écrit ou à voix haute : « Le canapé peut-il aller dans la chambre ? » Chaque réponse est vérifiée, et le plan des déplacements donne les étapes à imprimer.",
    chooseRoom: "Choisir une pièce", language: "Langue",
    pointCloud: "Nuage de points", objectBoxes: "Boîtes des objets",
    objectVolumes: "Volumes des objets", walkableFloor: "Sol praticable",
    cutAbove: "Couper au-dessus de",
    moveTitle: "Déplacer un objet",
    moveHint: "Cliquez sur un objet, puis là où il doit aller : sous une table, sur un bureau " +
      "(les objets légers sont portés), n'importe où. Après un déplacement, le clic suivant " +
      "choisit un nouvel objet ; pour en changer avant, utilisez « Changer », Ctrl-clic ou " +
      "Échap. Chaque déplacement est planifié et vérifié sur la géométrie réelle avant d'avoir lieu.",
    object: "Objet", target: "Cible", rotation: "Rotation", degrees: "degrés", keep: "inchangée",
    none: "aucun", change: "Changer", check: "Vérifier", move: "Déplacer", undo: "Annuler",
    reset: "Réinitialiser",
    planButton: "Plan des déplacements (PDF)",
    planTitle: "Les déplacements faits jusqu'ici, en étapes numérotées et en plan, à enregistrer en PDF",
    makeRoom: "Faire de la place et déplacer",
    askTitle: "Demander à PIVOT",
    askHint: "Par exemple : « La chaise 5 peut-elle aller sous la table 72 ? », « Mets la " +
      "poubelle à côté de la porte 26 sans la bloquer. » Chaque réponse est vérifiée sur la géométrie.",
    askPlaceholder: "Votre demande…", send: "Envoyer", speak: "Parler", stop: "Arrêter",
    transcribing: "Transcription…", micTitle: "Cliquez, parlez, cliquez à nouveau",
    readAloud: "Lire les réponses à voix haute",
    meta: "{objects} objets · {points} points",
    planning: "Planification…", error: "Erreur", moved: "Déplacé", feasible: "Faisable",
    pathLine: "trajet {length} m, rotation {rotation}°",
    effortLine: "effort {effort} m, passage le plus serré {clearance}", overOneMeter: "> 1 m",
    notPossible: "Impossible",
    status_no_path: "aucun chemin", status_invalid_goal: "l'emplacement n'est pas libre",
    status_invalid_start: "l'objet est coincé",
    status_search_limit_reached: "limite de recherche atteinte",
    inFront: "Remarque : l'objet se trouve devant cet élément : {label} (#{id}).",
    adjusted: "Cible décalée de {cm} cm pour laisser de la place autour de l'objet.",
    lookingForRoom: "Recherche d'objets à pousser pour faire de la place…",
    undone: "Dernière modification annulée.", sceneReset: "Scène réinitialisée.",
    thinking: "Réflexion…", checkedWith: "Vérifié avec : ",
    budgetSpent: "La démo publique a épuisé son budget : lancez PIVOT vous-même (voir le README).",
    requestsLeft: "Démo publique : encore {n} demandes aujourd'hui.",
    requestLeft: "Démo publique : encore 1 demande aujourd'hui.",
    heardNothing: "Je n'ai rien entendu.", micUnavailable: "Micro indisponible : ",
  },
  de: {
    remove: "Entfernen", removed: "{name} aus dem Raum entfernt. „Rückgängig“ stellt es zurück.", removedWith: "Ebenfalls entfernt: {names}.",
    tourNext: "Weiter",
    tourSkip: "Überspringen",
    tourDone: "Los geht's!",
    tourHelp: "Rundgang erneut zeigen",
    tour1: "Willkommen bei PIVOT! Das ist Ihr Raum, aus einem Video in 3D nachgebaut. Ziehen zum Drehen, Mausrad zum Zoomen.",
    tour2: "Links: Raum wählen, Anzeige (Punkte, Objekte, Boden) und Sprache.",
    tour3: "Zum Verschieben ein Objekt anklicken, dann die Zielstelle. PIVOT prüft die echte Geometrie, bevor sich etwas bewegt: „Prüfen“ testet, „Verschieben“ führt aus, „Rückgängig“ nimmt es zurück.",
    tour4: "Oder einfach fragen, getippt oder gesprochen: „Passt das Sofa ins Schlafzimmer?“ Jede Antwort wird geprüft, und der Umzugsplan liefert die Schritte zum Ausdrucken.",
    chooseRoom: "Raum wählen", language: "Sprache",
    pointCloud: "Punktwolke", objectBoxes: "Objektboxen", objectVolumes: "Objektvolumen",
    walkableFloor: "Begehbarer Boden", cutAbove: "Schnitt über",
    moveTitle: "Objekt verschieben",
    moveHint: "Klicken Sie auf ein Objekt und dann auf die Stelle, an die es soll: unter einen " +
      "Tisch, auf einen Schreibtisch (leichte Objekte werden getragen), überallhin. Nach einer " +
      "Verschiebung wählt der nächste Klick ein neues Objekt; um es vorher zu wechseln, nutzen " +
      "Sie „Wechseln“, Strg-Klick oder Esc. Jede Verschiebung wird geplant und an der echten " +
      "Geometrie geprüft, bevor sie geschieht.",
    object: "Objekt", target: "Ziel", rotation: "Drehung", degrees: "Grad", keep: "beibehalten",
    none: "keins", change: "Wechseln", check: "Prüfen", move: "Verschieben", undo: "Rückgängig",
    reset: "Zurücksetzen",
    planButton: "Umzugsplan (PDF)",
    planTitle: "Die bisherigen Verschiebungen als nummerierte Schritte und Plan, als PDF zu speichern",
    makeRoom: "Platz schaffen & verschieben",
    askTitle: "PIVOT fragen",
    askHint: "Zum Beispiel: „Passt Stuhl 5 unter Tisch 72?“, „Stell den Mülleimer neben Tür 26, " +
      "ohne sie zu blockieren.“ Jede Antwort wird an der Geometrie geprüft.",
    askPlaceholder: "Ihre Anfrage…", send: "Senden", speak: "Sprechen", stop: "Stopp",
    transcribing: "Transkription…", micTitle: "Klicken, sprechen, erneut klicken",
    readAloud: "Antworten vorlesen",
    meta: "{objects} Objekte · {points} Punkte",
    planning: "Planung…", error: "Fehler", moved: "Verschoben", feasible: "Machbar",
    pathLine: "Weg {length} m, Drehung {rotation}°",
    effortLine: "Aufwand {effort} m, engste Stelle {clearance}", overOneMeter: "> 1 m",
    notPossible: "Nicht möglich",
    status_no_path: "kein Weg", status_invalid_goal: "der Platz ist nicht frei",
    status_invalid_start: "das Objekt steckt fest",
    status_search_limit_reached: "Suchgrenze erreicht",
    inFront: "Hinweis: Das Objekt steht vor diesem Element: {label} (#{id}).",
    adjusted: "Ziel um {cm} cm verschoben, um Platz um das Objekt zu lassen.",
    lookingForRoom: "Suche nach Objekten, die aus dem Weg geräumt werden können…",
    undone: "Letzte Änderung rückgängig gemacht.", sceneReset: "Szene zurückgesetzt.",
    thinking: "Denke nach…", checkedWith: "Geprüft mit: ",
    budgetSpent: "Die öffentliche Demo hat ihr Budget aufgebraucht: Starten Sie PIVOT selbst " +
      "(siehe README).",
    requestsLeft: "Öffentliche Demo: heute noch {n} Anfragen.",
    requestLeft: "Öffentliche Demo: heute noch 1 Anfrage.",
    heardNothing: "Ich habe nichts gehört.", micUnavailable: "Mikrofon nicht verfügbar: ",
  },
  es: {
    remove: "Quitar", removed: "{name} retirado de la habitación. «Deshacer» lo devuelve.", removedWith: "También retirado: {names}.",
    tourNext: "Siguiente",
    tourSkip: "Omitir",
    tourDone: "¡Vamos!",
    tourHelp: "Ver la visita de nuevo",
    tour1: "¡Bienvenido a PIVOT! Esta es tu habitación, reconstruida en 3D a partir de un vídeo. Arrastra para girar alrededor y usa la rueda para acercar.",
    tour2: "A la izquierda: elige la habitación, qué mostrar (puntos, objetos, suelo) y el idioma.",
    tour3: "Para mover algo, haz clic en ello y luego donde debe ir. PIVOT comprueba la geometría real antes de mover nada: «Comprobar» prueba, «Mover» lo hace, «Deshacer» vuelve atrás.",
    tour4: "O simplemente pregunta, escribiendo o hablando: «¿Cabe el sofá en el dormitorio?» Cada respuesta se comprueba, y el plan de mudanza da los pasos para imprimir.",
    chooseRoom: "Elegir una habitación", language: "Idioma",
    pointCloud: "Nube de puntos", objectBoxes: "Cajas de los objetos",
    objectVolumes: "Volúmenes de los objetos", walkableFloor: "Suelo transitable",
    cutAbove: "Cortar por encima de",
    moveTitle: "Mover un objeto",
    moveHint: "Haz clic en un objeto y luego donde debe ir: debajo de una mesa, sobre un " +
      "escritorio (los objetos ligeros se llevan en brazos), donde sea. Tras un movimiento, el " +
      "siguiente clic elige un objeto nuevo; para cambiarlo antes, usa «Cambiar», Ctrl-clic o " +
      "Esc. Cada movimiento se planifica y se comprueba con la geometría real antes de hacerse.",
    object: "Objeto", target: "Destino", rotation: "Rotación", degrees: "grados", keep: "igual",
    none: "ninguno", change: "Cambiar", check: "Comprobar", move: "Mover", undo: "Deshacer",
    reset: "Restablecer",
    planButton: "Plan de mudanza (PDF)",
    planTitle: "Los movimientos hechos hasta ahora, en pasos numerados y un plano, para guardar en PDF",
    makeRoom: "Hacer sitio y mover",
    askTitle: "Pregunta a PIVOT",
    askHint: "Por ejemplo: «¿Cabe la silla 5 debajo de la mesa 72?», «Pon la papelera junto a la " +
      "puerta 26 sin bloquearla.» Cada respuesta se comprueba con la geometría.",
    askPlaceholder: "Tu petición…", send: "Enviar", speak: "Hablar", stop: "Parar",
    transcribing: "Transcribiendo…", micTitle: "Haz clic, habla, vuelve a hacer clic",
    readAloud: "Leer las respuestas en voz alta",
    meta: "{objects} objetos · {points} puntos",
    planning: "Planificando…", error: "Error", moved: "Movido", feasible: "Posible",
    pathLine: "recorrido {length} m, rotación {rotation}°",
    effortLine: "esfuerzo {effort} m, paso más estrecho {clearance}", overOneMeter: "> 1 m",
    notPossible: "No es posible",
    status_no_path: "no hay camino", status_invalid_goal: "el sitio no está libre",
    status_invalid_start: "el objeto está atascado",
    status_search_limit_reached: "límite de búsqueda alcanzado",
    inFront: "Nota: el objeto queda delante de este elemento: {label} (#{id}).",
    adjusted: "Destino desplazado {cm} cm para dejar espacio alrededor del objeto.",
    lookingForRoom: "Buscando objetos que apartar…",
    undone: "Último cambio deshecho.", sceneReset: "Escena restablecida.",
    thinking: "Pensando…", checkedWith: "Comprobado con: ",
    budgetSpent: "La demo pública ha agotado su presupuesto: ejecuta PIVOT tú mismo (ver el README).",
    requestsLeft: "Demo pública: quedan {n} peticiones hoy.",
    requestLeft: "Demo pública: queda 1 petición hoy.",
    heardNothing: "No he oído nada.", micUnavailable: "Micrófono no disponible: ",
  },
  it: {
    remove: "Rimuovi", removed: "{name} tolto dalla stanza. «Annulla» lo rimette.", removedWith: "Tolto anche: {names}.",
    tourNext: "Avanti",
    tourSkip: "Salta",
    tourDone: "Iniziamo!",
    tourHelp: "Rivedi la visita guidata",
    tour1: "Benvenuto in PIVOT! Questa è la tua stanza, ricostruita in 3D da un video. Trascina per girarci intorno, usa la rotella per lo zoom.",
    tour2: "A sinistra: scegli la stanza, cosa mostrare (punti, oggetti, pavimento) e la lingua.",
    tour3: "Per spostare qualcosa, cliccaci sopra, poi dove deve andare. PIVOT verifica la geometria reale prima di muovere qualsiasi cosa: «Verifica» prova, «Sposta» esegue, «Annulla» torna indietro.",
    tour4: "Oppure chiedi semplicemente, scrivendo o a voce: «Il divano entra in camera da letto?» Ogni risposta è verificata, e il piano del trasloco dà i passi da stampare.",
    chooseRoom: "Scegli una stanza", language: "Lingua",
    pointCloud: "Nuvola di punti", objectBoxes: "Riquadri degli oggetti",
    objectVolumes: "Volumi degli oggetti", walkableFloor: "Pavimento calpestabile",
    cutAbove: "Taglia sopra",
    moveTitle: "Sposta un oggetto",
    moveHint: "Clicca su un oggetto, poi dove deve andare: sotto un tavolo, su una scrivania (gli " +
      "oggetti leggeri vengono portati), ovunque. Dopo uno spostamento, il clic successivo sceglie " +
      "un nuovo oggetto; per cambiarlo prima, usa «Cambia», Ctrl-clic o Esc. Ogni spostamento è " +
      "pianificato e verificato sulla geometria reale prima di avvenire.",
    object: "Oggetto", target: "Destinazione", rotation: "Rotazione", degrees: "gradi",
    keep: "invariata", none: "nessuno", change: "Cambia", check: "Verifica", move: "Sposta",
    undo: "Annulla", reset: "Ripristina",
    planButton: "Piano del trasloco (PDF)",
    planTitle: "Gli spostamenti fatti finora, in passi numerati e una pianta, da salvare in PDF",
    makeRoom: "Fai spazio e sposta",
    askTitle: "Chiedi a PIVOT",
    askHint: "Per esempio: «La sedia 5 entra sotto il tavolo 72?», «Metti il cestino accanto alla " +
      "porta 26 senza bloccarla.» Ogni risposta è verificata sulla geometria.",
    askPlaceholder: "La tua richiesta…", send: "Invia", speak: "Parla", stop: "Stop",
    transcribing: "Trascrizione…", micTitle: "Clicca, parla, clicca di nuovo",
    readAloud: "Leggi le risposte ad alta voce",
    meta: "{objects} oggetti · {points} punti",
    planning: "Pianificazione…", error: "Errore", moved: "Spostato", feasible: "Fattibile",
    pathLine: "percorso {length} m, rotazione {rotation}°",
    effortLine: "sforzo {effort} m, passaggio più stretto {clearance}", overOneMeter: "> 1 m",
    notPossible: "Non è possibile",
    status_no_path: "nessun percorso", status_invalid_goal: "il posto non è libero",
    status_invalid_start: "l'oggetto è bloccato",
    status_search_limit_reached: "limite di ricerca raggiunto",
    inFront: "Nota: l'oggetto si trova davanti a questo elemento: {label} (#{id}).",
    adjusted: "Destinazione spostata di {cm} cm per lasciare spazio intorno all'oggetto.",
    lookingForRoom: "Cerco oggetti da spostare per fare spazio…",
    undone: "Ultima modifica annullata.", sceneReset: "Scena ripristinata.",
    thinking: "Sto pensando…", checkedWith: "Verificato con: ",
    budgetSpent: "La demo pubblica ha esaurito il budget: esegui PIVOT da solo (vedi il README).",
    requestsLeft: "Demo pubblica: ancora {n} richieste oggi.",
    requestLeft: "Demo pubblica: ancora 1 richiesta oggi.",
    heardNothing: "Non ho sentito nulla.", micUnavailable: "Microfono non disponibile: ",
  },
  zh: {
    remove: "移除", removed: "已将{name}移出房间。“撤销”可将其放回。", removedWith: "一并移出：{names}。",
    tourNext: "下一步",
    tourSkip: "跳过",
    tourDone: "开始吧！",
    tourHelp: "重新查看引导",
    tour1: "欢迎使用 PIVOT！这是根据视频重建的 3D 房间。拖动可环绕查看，滚轮可缩放。",
    tour2: "左侧：选择房间、显示内容（点云、物体、地面）和语言。",
    tour3: "要移动物体，先点击它，再点击目标位置。PIVOT 会在移动前检查真实几何：“检查”只做测试，“移动”执行，“撤销”返回上一步。",
    tour4: "也可以直接用文字或语音提问：“沙发能搬进卧室吗？”每个回答都会经过检查，搬运计划会给出可打印的步骤。",
    chooseRoom: "选择房间", language: "语言",
    pointCloud: "点云", objectBoxes: "物体边框", objectVolumes: "物体体积",
    walkableFloor: "可通行地面", cutAbove: "剖切高度",
    moveTitle: "移动物体",
    moveHint: "点击一个物体，再点击它要去的位置：桌子下面、书桌上（轻的物体会被搬起）、任何地方。" +
      "移动之后，下一次点击会选择新的物体；若要提前更换，请使用“更换”、Ctrl+点击或 Esc。" +
      "每次移动在执行前都会根据真实几何进行规划和检查。",
    object: "物体", target: "目标", rotation: "旋转", degrees: "度", keep: "不变",
    none: "无", change: "更换", check: "检查", move: "移动", undo: "撤销", reset: "重置",
    planButton: "搬运计划（PDF）",
    planTitle: "目前已完成的移动，以编号步骤和平面图呈现，可保存为 PDF",
    makeRoom: "腾出空间并移动",
    askTitle: "询问 PIVOT",
    askHint: "例如：“5 号椅子能放到 72 号桌子下面吗？”“把垃圾桶放到 26 号门旁边，但不要挡住门。”" +
      "每个回答都会根据几何进行检查。",
    askPlaceholder: "你的请求…", send: "发送", speak: "说话", stop: "停止",
    transcribing: "正在转写…", micTitle: "点击，说话，再点击一次", readAloud: "朗读回答",
    meta: "{objects} 个物体 · {points} 个点",
    planning: "正在规划…", error: "错误", moved: "已移动", feasible: "可行",
    pathLine: "路径 {length} 米，旋转 {rotation}°",
    effortLine: "工作量 {effort} 米，最窄处间隙 {clearance}", overOneMeter: "> 1 米",
    notPossible: "无法完成",
    status_no_path: "没有路径", status_invalid_goal: "该位置被占用",
    status_invalid_start: "物体被卡住", status_search_limit_reached: "达到搜索上限",
    inFront: "注意：该物体挡在{label}（#{id}）前面。",
    adjusted: "目标已移动 {cm} 厘米，以便在物体周围留出空间。",
    lookingForRoom: "正在寻找可以挪开的物体…",
    undone: "已撤销上一步操作。", sceneReset: "场景已重置。",
    thinking: "思考中…", checkedWith: "检查工具：",
    budgetSpent: "公开演示的预算已用完：请自行运行 PIVOT（见 README）。",
    requestsLeft: "公开演示：今天还剩 {n} 次请求。",
    requestLeft: "公开演示：今天还剩 1 次请求。",
    heardNothing: "我什么也没听到。", micUnavailable: "麦克风不可用：",
  },
};

// Names of objects and rooms: French, German, Spanish, Italian, Chinese.
const NAMES = {
  "chair": ["chaise", "Stuhl", "silla", "sedia", "椅子"],
  "office chair": ["chaise de bureau", "Bürostuhl", "silla de oficina", "sedia da ufficio", "办公椅"],
  "armchair": ["fauteuil", "Sessel", "sillón", "poltrona", "扶手椅"],
  "stool": ["tabouret", "Hocker", "taburete", "sgabello", "凳子"],
  "sofa": ["canapé", "Sofa", "sofá", "divano", "沙发"],
  "couch": ["canapé", "Couch", "sofá", "divano", "沙发"],
  "bed": ["lit", "Bett", "cama", "letto", "床"],
  "table": ["table", "Tisch", "mesa", "tavolo", "桌子"],
  "coffee table": ["table basse", "Couchtisch", "mesa de centro", "tavolino", "茶几"],
  "dining table": ["table à manger", "Esstisch", "mesa de comedor", "tavolo da pranzo", "餐桌"],
  "desk": ["bureau", "Schreibtisch", "escritorio", "scrivania", "书桌"],
  "dresser": ["commode", "Kommode", "cómoda", "cassettiera", "五斗柜"],
  "nightstand": ["table de chevet", "Nachttisch", "mesita de noche", "comodino", "床头柜"],
  "cabinet": ["meuble", "Schrank", "armario", "mobile", "柜子"],
  "wardrobe": ["armoire", "Kleiderschrank", "armario ropero", "armadio", "衣柜"],
  "fitted wardrobe": ["placard", "Einbauschrank", "armario empotrado", "armadio a muro", "嵌入式衣柜"],
  "shelf": ["étagère", "Regal", "estantería", "scaffale", "架子"],
  "bookshelf": ["bibliothèque", "Bücherregal", "librería", "libreria", "书架"],
  "sideboard": ["buffet", "Sideboard", "aparador", "credenza", "餐边柜"],
  "tv stand": ["meuble TV", "TV-Möbel", "mueble de TV", "mobile TV", "电视柜"],
  "television": ["télévision", "Fernseher", "televisor", "televisore", "电视"],
  "tv": ["télévision", "Fernseher", "televisor", "televisore", "电视"],
  "monitor": ["écran", "Monitor", "monitor", "monitor", "显示器"],
  "lamp": ["lampe", "Lampe", "lámpara", "lampada", "灯"],
  "floor lamp": ["lampadaire", "Stehlampe", "lámpara de pie", "lampada da terra", "落地灯"],
  "ceiling lamp": ["plafonnier", "Deckenlampe", "lámpara de techo", "lampada da soffitto", "吊灯"],
  "rug": ["tapis", "Teppich", "alfombra", "tappeto", "地毯"],
  "carpet": ["tapis", "Teppich", "alfombra", "tappeto", "地毯"],
  "mat": ["tapis", "Matte", "alfombrilla", "tappetino", "垫子"],
  "plant": ["plante", "Pflanze", "planta", "pianta", "植物"],
  "mirror": ["miroir", "Spiegel", "espejo", "specchio", "镜子"],
  "painting": ["tableau", "Gemälde", "cuadro", "quadro", "画"],
  "picture": ["tableau", "Bild", "cuadro", "quadro", "画"],
  "poster": ["affiche", "Poster", "póster", "poster", "海报"],
  "curtain": ["rideau", "Vorhang", "cortina", "tenda", "窗帘"],
  "blinds": ["store", "Jalousie", "persiana", "veneziana", "百叶窗"],
  "door": ["porte", "Tür", "puerta", "porta", "门"],
  "window": ["fenêtre", "Fenster", "ventana", "finestra", "窗户"],
  "trash can": ["poubelle", "Mülleimer", "papelera", "cestino", "垃圾桶"],
  "bin": ["poubelle", "Mülleimer", "papelera", "cestino", "垃圾桶"],
  "box": ["carton", "Karton", "caja", "scatola", "箱子"],
  "carton": ["carton", "Karton", "caja", "scatola", "纸箱"],
  "bench": ["banc", "Bank", "banco", "panca", "长凳"],
  "kitchen counter": ["plan de travail", "Küchenzeile", "encimera", "piano di lavoro", "厨房台面"],
  "kitchen island": ["îlot de cuisine", "Kücheninsel", "isla de cocina", "isola della cucina", "厨房岛台"],
  "counter": ["comptoir", "Theke", "mostrador", "bancone", "柜台"],
  "bathtub": ["baignoire", "Badewanne", "bañera", "vasca da bagno", "浴缸"],
  "shower": ["douche", "Dusche", "ducha", "doccia", "淋浴"],
  "toilet": ["toilettes", "Toilette", "inodoro", "water", "马桶"],
  "sink": ["évier", "Spüle", "fregadero", "lavello", "水槽"],
  "fridge": ["réfrigérateur", "Kühlschrank", "nevera", "frigorifero", "冰箱"],
  "fireplace": ["cheminée", "Kamin", "chimenea", "camino", "壁炉"],
  "radiator": ["radiateur", "Heizkörper", "radiador", "termosifone", "暖气片"],
  "column": ["colonne", "Säule", "columna", "colonna", "柱子"],
  "whiteboard": ["tableau blanc", "Whiteboard", "pizarra blanca", "lavagna bianca", "白板"],
  "board": ["tableau", "Tafel", "pizarra", "lavagna", "黑板"],
  "backpack": ["sac à dos", "Rucksack", "mochila", "zaino", "背包"],
  "bag": ["sac", "Tasche", "bolsa", "borsa", "包"],
  "vase": ["vase", "Vase", "jarrón", "vaso", "花瓶"],
  "living room": ["salon", "Wohnzimmer", "salón", "soggiorno", "客厅"],
  "bedroom": ["chambre", "Schlafzimmer", "dormitorio", "camera da letto", "卧室"],
  "hallway": ["entrée", "Flur", "recibidor", "ingresso", "走廊"],
  "kitchen": ["cuisine", "Küche", "cocina", "cucina", "厨房"],
  "office": ["bureau", "Büro", "oficina", "ufficio", "办公室"],
  "bathroom": ["salle de bain", "Badezimmer", "baño", "bagno", "浴室"],
  "dining room": ["salle à manger", "Esszimmer", "comedor", "sala da pranzo", "餐厅"],
  "classroom": ["salle de classe", "Klassenzimmer", "aula", "aula", "教室"],
  "room": ["pièce", "Raum", "habitación", "stanza", "房间"],
  "apartment": ["appartement", "Wohnung", "apartamento", "appartamento", "公寓"],
};
const NAME_COLUMNS = { fr: 0, de: 1, es: 2, it: 3, zh: 4 };

const STORAGE_KEY = "pivot.language";

function stored() {
  try { return localStorage.getItem(STORAGE_KEY); } catch { return null; }
}

// The language to show: the one chosen here before, else the browser's first
// supported one, else English.
function initialLanguage(preferred = navigator.languages || [navigator.language]) {
  const chosen = stored();
  if (chosen && TEXTS[chosen]) return chosen;
  for (const tag of preferred) {
    const code = String(tag || "").toLowerCase().split("-")[0];
    if (TEXTS[code]) return code;
  }
  return "en";
}

function remember(language) {
  try { localStorage.setItem(STORAGE_KEY, language); } catch { /* Private mode: not kept. */ }
}

// A text of the page, with {placeholders} filled in.
function text(language, key, values = {}) {
  const template = (TEXTS[language] || TEXTS.en)[key] ?? TEXTS.en[key] ?? key;
  return template.replace(/\{(\w+)\}/g, (_, name) => String(values[name] ?? ""));
}

// The name of an object or a room ("bedroom 2" keeps its number), as it is
// when the language has no word for it. Capitalized like the original.
function name(language, english) {
  const column = NAME_COLUMNS[language];
  if (column === undefined || !english) return english;
  const match = /^(.*?)(\s+\d+)?$/.exec(english.trim());
  const base = match[1].toLowerCase();
  const words = NAMES[base];
  if (!words) return english;
  let translated = words[column];
  if (english[0] !== english[0].toLowerCase()) {
    translated = translated[0].toUpperCase() + translated.slice(1);
  }
  return translated + (match[2] || "");
}

window.PivotI18n = { LANGUAGES, initialLanguage, remember, text, name };
})();
