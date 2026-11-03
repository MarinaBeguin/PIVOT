# Demo checklist

What to try on the live demo before a release or a recording, in order. Each
line is one check: what to do, then what must happen. Run it twice: on a laptop
(Chrome or Firefox), then on a phone (Safari on iOS, Chrome on Android).

## Opening

On a phone, use the HTTPS address of `modal serve` (or the deployed demo): the microphone needs HTTPS.

- [ ] Open the demo URL in a private window: the office loads in under 10 s, the tour starts.
- [ ] Tour: every bubble sits inside the screen, next to what it describes, and never covers the panel; "Next" reaches the last step; on a phone, the texts say "tap", not "click".
- [ ] Close the tour, reload: it does not start again; the "?" button starts it again.
- [ ] Rooms menu: office, bedroom, classroom, apartment each load; the camera frames the whole room.
- [ ] Display: points, boxes, labels, floor, volumes each switch on and off; the cut slider hides the ceiling.

## Languages

- [ ] Language menu: English, Français, Deutsch, Español, Italiano, 中文. Every text of the panel, the tour and the chat changes, and nothing stays in English.
- [ ] Reload: the language chosen is kept.

## Moving objects by hand

- [ ] Click a chair, then a free spot: it slides there, around the other furniture; the result line says how far and by which way.
- [ ] Click a spot inside a wall, through a window or on top of a desk: refused, with the reason.
- [ ] Rotation: change the angle, move again: the chair ends turned.
- [ ] A rug: it can be moved; furniture standing on it is moved off first, or it is rolled.
- [ ] A picture on a wall: it moves along walls only, flat against them.
- [ ] Undo, then Reset: back to the previous step, then to the room as filmed.
- [ ] "Make room here": pick a spot taken by furniture; the furniture in the way moves aside, the fewest objects possible, and the spot is free.

## Removing and adding

- [ ] Remove an object: it disappears; Undo brings it back.
- [ ] "Add from a photo": photograph a chair (phone camera, or a file): its cutout appears as a ghost that follows the pointer; it is pinned on click (tap on a phone); right-click drops it.
- [ ] A photo with several objects: a popup asks which one.
- [ ] The size dialog: the proposed size is plausible; changing it changes the ghost.
- [ ] A spot under another object (under a desk): refused for the added object; on a rug: accepted.

## Asking PIVOT

- [ ] "Move the chair next to the window": PIVOT answers in a few seconds, the chair moves, the answer names the checks made.
- [ ] An impossible request ("put the wardrobe on the desk"): PIVOT says it cannot, and why, and nothing moves.
- [ ] Wheelchair: "Make room for a wheelchair at table 24": a 90 cm passage from the door and a free spot at the table; the floor a wheelchair reaches is shaded.
- [ ] A request in French, then in Chinese: answers in the same language.
- [ ] Microphone: speak a request; it is transcribed into the box.
- [ ] "Read the answers aloud": a voice speaks at once, then reads each answer; with `PIVOT_TTS=mms`, the same voice on every device (English, French, German, Spanish), Safari included.
- [ ] "Will an IKEA KIVIK 3-seat sofa fit next to the window?": PIVOT looks its size up on the web and names the page.
- [ ] A link to a Marketplace ad: PIVOT reads the size from the ad (or asks for a screenshot of it, which "Add from a photo" reads).
- [ ] A long request ("make room for a sofa under the window"): "Stop" ends it within a second or two; the moves already made stay, Undo takes them back.
- [ ] Limits: the counter of requests left goes down; at 0, a calm message (not red) says when it comes back.

## Moving plan

- [ ] After a few moves, "Moving plan (PDF)": the page lists the steps in order, with distances, turns and two-person moves; objects removed come first, objects added last.
- [ ] Each path drawn on the plan is the one the 3D view played (same bends around the furniture).
- [ ] The plan is in the language chosen (try French and Chinese).
- [ ] Objects added from a photo appear in the plan, in green.
- [ ] "Save as PDF or print" opens the print dialog; the PDF fits on A4.

## Your own video (only where enabled)

- [ ] On the public demo (`PIVOT_UPLOADS` unset): the "Send your video" button is not shown.
- [ ] With `PIVOT_UPLOADS=1` on a public server: sending without the owner key is refused (403).
- [ ] With the owner key, or locally with `PIVOT_UPLOADS=1`: send a short phone video; the status says it is building and shows the last log lines; once ready, the new room is in the rooms menu.
- [ ] The room built: no face, plate or name readable in the points.

## Phone only

- [ ] The panel is a drawer: it opens and closes by swiping; the 3D view stays usable.
- [ ] Object numbers are readable; tapping an object selects it.
- [ ] Pinch zoom and two-finger pan move the camera, never the page.
