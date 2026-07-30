// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

export const PROMPT_EXAMPLE_EVENT = "ardy:prompt-example-selected"

export const DEFAULT_PROMPT =
  "A person walks forward, then waves with their right hand."

export const PROMPT_EXAMPLE_CATEGORIES = [
  "Locomotion",
  "Direction changes",
  "Jumps and agility",
  "Gestures",
  "Dance and rhythm",
  "Exercise and sports",
  "Stances and transitions",
  "Balance and coordination",
  "Pantomime",
  "Motion sequences",
] as const

export type PromptExampleCategory =
  (typeof PROMPT_EXAMPLE_CATEGORIES)[number]

export interface PromptExample {
  readonly label: string
  readonly prompt: string
  readonly category: PromptExampleCategory
}

export const PROMPT_EXAMPLES = [
  {
    label: "Walk and wave",
    prompt: DEFAULT_PROMPT,
    category: "Locomotion",
  },
  {
    label: "Confident walk",
    prompt: "A person walks forward confidently with relaxed arm swings.",
    category: "Locomotion",
  },
  {
    label: "Slow walk",
    prompt: "A person walks forward slowly with careful, measured steps.",
    category: "Locomotion",
  },
  {
    label: "Brisk walk",
    prompt: "A person walks forward briskly with energetic arm swings.",
    category: "Locomotion",
  },
  {
    label: "Steady run",
    prompt: "A person runs forward at a steady pace.",
    category: "Locomotion",
  },
  {
    label: "Light jog",
    prompt: "A person jogs forward lightly at a comfortable pace.",
    category: "Locomotion",
  },
  {
    label: "Backward walk",
    prompt: "A person walks backward carefully while looking ahead.",
    category: "Locomotion",
  },
  {
    label: "Shuffle right",
    prompt: "A person shuffles sideways to the right with quick steps.",
    category: "Locomotion",
  },
  {
    label: "Shuffle left",
    prompt: "A person shuffles sideways to the left with quick steps.",
    category: "Locomotion",
  },
  {
    label: "Tiptoe forward",
    prompt: "A person tiptoes forward quietly with small, cautious steps.",
    category: "Locomotion",
  },
  {
    label: "Turn left",
    prompt: "A person turns left and continues walking forward.",
    category: "Direction changes",
  },
  {
    label: "Turn right",
    prompt: "A person turns right and continues walking forward.",
    category: "Direction changes",
  },
  {
    label: "About-face",
    prompt: "A person makes a smooth half turn and faces the opposite direction.",
    category: "Direction changes",
  },
  {
    label: "Clockwise circle",
    prompt: "A person walks clockwise in a small circle.",
    category: "Direction changes",
  },
  {
    label: "Counterclockwise circle",
    prompt: "A person walks counterclockwise in a small circle.",
    category: "Direction changes",
  },
  {
    label: "Zigzag walk",
    prompt: "A person walks forward in a gentle zigzag pattern.",
    category: "Direction changes",
  },
  {
    label: "Pivot left",
    prompt: "A person plants both feet and pivots sharply to the left.",
    category: "Direction changes",
  },
  {
    label: "Pivot right",
    prompt: "A person plants both feet and pivots sharply to the right.",
    category: "Direction changes",
  },
  {
    label: "Diagonal walk",
    prompt: "A person walks diagonally forward and to the right.",
    category: "Direction changes",
  },
  {
    label: "Figure-eight walk",
    prompt: "A person walks in a compact figure-eight pattern.",
    category: "Direction changes",
  },
  {
    label: "Vertical jump",
    prompt: "A person jumps straight upward and lands on both feet.",
    category: "Jumps and agility",
  },
  {
    label: "Forward hop",
    prompt: "A person hops forward with both feet and lands steadily.",
    category: "Jumps and agility",
  },
  {
    label: "Backward hop",
    prompt: "A person hops backward with both feet and regains balance.",
    category: "Jumps and agility",
  },
  {
    label: "Hop left",
    prompt: "A person hops sideways to the left and lands on both feet.",
    category: "Jumps and agility",
  },
  {
    label: "Hop right",
    prompt: "A person hops sideways to the right and lands on both feet.",
    category: "Jumps and agility",
  },
  {
    label: "Jumping jacks",
    prompt: "A person performs several energetic jumping jacks.",
    category: "Jumps and agility",
  },
  {
    label: "Skip forward",
    prompt: "A person skips forward with a light, playful rhythm.",
    category: "Jumps and agility",
  },
  {
    label: "Side gallop",
    prompt: "A person gallops sideways to the right with springy steps.",
    category: "Jumps and agility",
  },
  {
    label: "High knees",
    prompt: "A person runs in place while lifting their knees high.",
    category: "Jumps and agility",
  },
  {
    label: "Two-foot bounce",
    prompt: "A person makes several small, rhythmic bounces on both feet.",
    category: "Jumps and agility",
  },
  {
    label: "Wave right",
    prompt: "A person stands still and waves with their right hand.",
    category: "Gestures",
  },
  {
    label: "Wave left",
    prompt: "A person stands still and waves with their left hand.",
    category: "Gestures",
  },
  {
    label: "Wave both hands",
    prompt: "A person raises both arms and waves both hands enthusiastically.",
    category: "Gestures",
  },
  {
    label: "Point right",
    prompt: "A person extends their right arm and points to the right.",
    category: "Gestures",
  },
  {
    label: "Point left",
    prompt: "A person extends their left arm and points to the left.",
    category: "Gestures",
  },
  {
    label: "Thumbs up",
    prompt: "A person raises their right hand and gives a clear thumbs-up gesture.",
    category: "Gestures",
  },
  {
    label: "Clap",
    prompt: "A person claps their hands together several times.",
    category: "Gestures",
  },
  {
    label: "Polite bow",
    prompt: "A person bends forward in a slow, polite bow and rises.",
    category: "Gestures",
  },
  {
    label: "Salute",
    prompt: "A person raises their right hand and gives a crisp salute.",
    category: "Gestures",
  },
  {
    label: "Shrug",
    prompt: "A person lifts both shoulders and turns their palms upward in a shrug.",
    category: "Gestures",
  },
  {
    label: "Joyful dance",
    prompt: "A person performs a joyful dance.",
    category: "Dance and rhythm",
  },
  {
    label: "Simple groove",
    prompt: "A person steps in place with a relaxed, rhythmic groove.",
    category: "Dance and rhythm",
  },
  {
    label: "Spin dance",
    prompt: "A person dances in place and completes a smooth full spin.",
    category: "Dance and rhythm",
  },
  {
    label: "Two-step dance",
    prompt: "A person performs a simple two-step dance from side to side.",
    category: "Dance and rhythm",
  },
  {
    label: "Hip sway",
    prompt: "A person sways their hips from side to side in rhythm.",
    category: "Dance and rhythm",
  },
  {
    label: "Arm wave dance",
    prompt: "A person performs a flowing dance wave through both arms.",
    category: "Dance and rhythm",
  },
  {
    label: "Salsa basic",
    prompt: "A person performs a basic salsa step with lively hip movement.",
    category: "Dance and rhythm",
  },
  {
    label: "Robot dance",
    prompt: "A person performs a stiff, mechanical robot dance.",
    category: "Dance and rhythm",
  },
  {
    label: "Ballet turn",
    prompt: "A person rises lightly and performs a controlled ballet turn.",
    category: "Dance and rhythm",
  },
  {
    label: "Victory dance",
    prompt: "A person performs a short celebratory victory dance.",
    category: "Dance and rhythm",
  },
  {
    label: "Deep squat",
    prompt: "A person lowers into a deep squat and returns to standing.",
    category: "Exercise and sports",
  },
  {
    label: "Right lunge",
    prompt: "A person steps forward with the right leg into a controlled lunge.",
    category: "Exercise and sports",
  },
  {
    label: "Left lunge",
    prompt: "A person steps forward with the left leg into a controlled lunge.",
    category: "Exercise and sports",
  },
  {
    label: "Push-up",
    prompt: "A person lowers to the floor and performs a single push-up.",
    category: "Exercise and sports",
  },
  {
    label: "Sit-up",
    prompt: "A person lies on their back and performs a controlled sit-up.",
    category: "Exercise and sports",
  },
  {
    label: "Shoulder stretch",
    prompt: "A person stretches one arm across their chest and switches sides.",
    category: "Exercise and sports",
  },
  {
    label: "Right front kick",
    prompt: "A person steps forward and delivers a controlled kick with the right foot.",
    category: "Exercise and sports",
  },
  {
    label: "Left front kick",
    prompt: "A person steps forward and delivers a controlled kick with the left foot.",
    category: "Exercise and sports",
  },
  {
    label: "Boxing combination",
    prompt: "A person throws a quick left jab followed by a right cross.",
    category: "Exercise and sports",
  },
  {
    label: "Overhead throw",
    prompt: "A person winds up and performs a strong overhead throwing motion.",
    category: "Exercise and sports",
  },
  {
    label: "Crouch and rise",
    prompt: "A person crouches down low and stands back up.",
    category: "Stances and transitions",
  },
  {
    label: "Right-knee kneel",
    prompt: "A person lowers carefully onto their right knee.",
    category: "Stances and transitions",
  },
  {
    label: "Left-knee kneel",
    prompt: "A person lowers carefully onto their left knee.",
    category: "Stances and transitions",
  },
  {
    label: "Sit on floor",
    prompt: "A person lowers from standing into a seated position on the floor.",
    category: "Stances and transitions",
  },
  {
    label: "Stand from floor",
    prompt: "A person rises smoothly from the floor to a standing position.",
    category: "Stances and transitions",
  },
  {
    label: "Lie on back",
    prompt: "A person lowers carefully to the floor and lies on their back.",
    category: "Stances and transitions",
  },
  {
    label: "Cross-legged sit",
    prompt: "A person sits down and settles into a cross-legged position.",
    category: "Stances and transitions",
  },
  {
    label: "Hands on hips",
    prompt: "A person stands upright with both hands resting on their hips.",
    category: "Stances and transitions",
  },
  {
    label: "Guard stance",
    prompt: "A person steps into a balanced fighting guard stance.",
    category: "Stances and transitions",
  },
  {
    label: "Relaxed idle",
    prompt: "A person stands comfortably and shifts their weight slightly.",
    category: "Stances and transitions",
  },
  {
    label: "Balance on right foot",
    prompt: "A person balances steadily on their right foot.",
    category: "Balance and coordination",
  },
  {
    label: "Balance on left foot",
    prompt: "A person balances steadily on their left foot.",
    category: "Balance and coordination",
  },
  {
    label: "Tree pose",
    prompt: "A person holds a calm tree pose with both hands overhead.",
    category: "Balance and coordination",
  },
  {
    label: "Heel-to-toe walk",
    prompt: "A person walks forward slowly, placing each heel directly before the other toe.",
    category: "Balance and coordination",
  },
  {
    label: "Airplane balance",
    prompt: "A person leans forward on one leg with both arms extended sideways.",
    category: "Balance and coordination",
  },
  {
    label: "Reach to toes",
    prompt: "A person bends forward and reaches both hands toward their toes.",
    category: "Balance and coordination",
  },
  {
    label: "Cross-body taps",
    prompt: "A person alternates touching each hand to the opposite knee.",
    category: "Balance and coordination",
  },
  {
    label: "Arm circles",
    prompt: "A person extends both arms and makes wide forward circles.",
    category: "Balance and coordination",
  },
  {
    label: "Right leg swing",
    prompt: "A person balances on the left leg and swings the right leg gently.",
    category: "Balance and coordination",
  },
  {
    label: "Left leg swing",
    prompt: "A person balances on the right leg and swings the left leg gently.",
    category: "Balance and coordination",
  },
  {
    label: "Push heavy door",
    prompt: "A person leans forward and pretends to push a heavy door open.",
    category: "Pantomime",
  },
  {
    label: "Pull rope",
    prompt: "A person plants their feet and pretends to pull a heavy rope.",
    category: "Pantomime",
  },
  {
    label: "Lift box",
    prompt: "A person bends down and pretends to lift a heavy box.",
    category: "Pantomime",
  },
  {
    label: "Carry box",
    prompt: "A person walks forward while pretending to carry a box with both hands.",
    category: "Pantomime",
  },
  {
    label: "Row a boat",
    prompt: "A person sits and repeats a strong rowing motion with both arms.",
    category: "Pantomime",
  },
  {
    label: "Climb ladder",
    prompt: "A person pretends to climb a ladder using alternating hands and feet.",
    category: "Pantomime",
  },
  {
    label: "Dig ground",
    prompt: "A person bends forward and repeats a forceful digging motion.",
    category: "Pantomime",
  },
  {
    label: "Sweep floor",
    prompt: "A person pretends to sweep the floor with broad side-to-side strokes.",
    category: "Pantomime",
  },
  {
    label: "Hammer downward",
    prompt: "A person repeatedly swings an imaginary hammer downward.",
    category: "Pantomime",
  },
  {
    label: "Turn a wheel",
    prompt: "A person grips an imaginary large wheel and turns it with both hands.",
    category: "Pantomime",
  },
  {
    label: "Walk then stop",
    prompt: "A person walks forward several steps and comes to a clean stop.",
    category: "Motion sequences",
  },
  {
    label: "Run then stop",
    prompt: "A person runs forward briefly and slows to a controlled stop.",
    category: "Motion sequences",
  },
  {
    label: "Step and kick",
    prompt: "A person steps forward, kicks with the right foot, and returns to standing.",
    category: "Motion sequences",
  },
  {
    label: "Turn and point",
    prompt: "A person turns to the left and points forward with the right hand.",
    category: "Motion sequences",
  },
  {
    label: "Jump and wave",
    prompt: "A person jumps once, lands steadily, and waves with both hands.",
    category: "Motion sequences",
  },
  {
    label: "Crouch then jump",
    prompt: "A person crouches low, jumps upward, and lands on both feet.",
    category: "Motion sequences",
  },
  {
    label: "Back up and turn",
    prompt: "A person walks backward several steps and turns to face left.",
    category: "Motion sequences",
  },
  {
    label: "Side step and clap",
    prompt: "A person steps to the right twice and claps their hands.",
    category: "Motion sequences",
  },
  {
    label: "Jog and stretch",
    prompt: "A person jogs in place, stops, and reaches both arms overhead.",
    category: "Motion sequences",
  },
  {
    label: "Bow and wave",
    prompt: "A person gives a polite bow, rises, and waves with the right hand.",
    category: "Motion sequences",
  },
] as const satisfies readonly PromptExample[]

export const PROMPT_EXAMPLE_GROUPS = PROMPT_EXAMPLE_CATEGORIES.map(
  (category) => ({
    value: category,
    items: PROMPT_EXAMPLES.filter(
      (example) => example.category === category
    ),
  })
)

export function matchesPromptExample(
  example: PromptExample,
  query: string
): boolean {
  const terms = query.trim().toLocaleLowerCase("en-US").split(/\s+/u)
  if (terms.length === 1 && terms[0] === "") return true

  const searchableText =
    `${example.label} ${example.prompt} ${example.category}`.toLocaleLowerCase(
      "en-US"
    )
  return terms.every((term) => searchableText.includes(term))
}
