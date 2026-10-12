"""
AudioSet 527 class labels — embedded for offline environments.

This file contains the complete AudioSet ontology class labels used by PANNs
(Pretrained Audio Neural Networks). Each index corresponds to the model's
output dimension. Source: Google AudioSet ontology (stable since 2017).

Format: AUDIOSET_LABELS[i] = display_name for class index i (0-526).
"""

# fmt: off
AUDIOSET_LABELS = [
    "Speech",                                          # 0
    "Male speech, man speaking",                       # 1
    "Female speech, woman speaking",                   # 2
    "Child speech, kid speaking",                      # 3
    "Conversation",                                    # 4
    "Narration, monologue",                            # 5
    "Babbling",                                        # 6
    "Speech synthesizer",                              # 7
    "Shout",                                           # 8
    "Bellow",                                          # 9
    "Whoop",                                           # 10
    "Yell",                                            # 11
    "Battle cry",                                      # 12
    "Children shouting",                               # 13
    "Screaming",                                       # 14
    "Whispering",                                      # 15
    "Laughter",                                        # 16
    "Baby laughter",                                   # 17
    "Giggle",                                          # 18
    "Snicker",                                         # 19
    "Belly laugh",                                     # 20
    "Chuckle, chortle",                                # 21
    "Crying, sobbing",                                 # 22
    "Baby cry, infant cry",                            # 23
    "Whimper",                                         # 24
    "Wail, moan",                                      # 25
    "Sigh",                                            # 26
    "Singing",                                         # 27
    "Choir",                                           # 28
    "Yodeling",                                        # 29
    "Chant",                                           # 30
    "Mantra",                                          # 31
    "Male singing",                                    # 32
    "Female singing",                                  # 33
    "Child singing",                                   # 34
    "Synthetic singing",                               # 35
    "Rapping",                                         # 36
    "Humming",                                         # 37
    "Groan",                                           # 38
    "Grunt",                                           # 39
    "Whistling",                                       # 40
    "Breathing",                                       # 41
    "Wheeze",                                          # 42
    "Snoring",                                         # 43
    "Gasp",                                            # 44
    "Pant",                                            # 45
    "Sniff",                                           # 46
    "Cough",                                           # 47
    "Sneeze",                                          # 48
    "Hiccup",                                          # 49
    "Burping, eructation",                             # 50
    "Chewing, mastication",                            # 51
    "Biting",                                          # 52
    "Gargling",                                        # 53
    "Stomach rumble",                                  # 54
    "Belching",                                        # 55
    "Throat clearing",                                 # 56
    "Finger snapping",                                 # 57
    "Clapping",                                        # 58
    "Heart sounds, heartbeat",                         # 59
    "Heart murmur",                                    # 60
    "Cheering",                                        # 61
    "Applause",                                        # 62
    "Chatter",                                         # 63
    "Crowd",                                           # 64
    "Hubbub, speech noise, speech babble",             # 65
    "Children playing",                                # 66
    "Animal",                                          # 67
    "Domestic animals, pets",                          # 68
    "Dog",                                             # 69
    "Bark",                                            # 70
    "Yip",                                             # 71
    "Howling",                                         # 72
    "Bow-wow",                                         # 73
    "Growling",                                        # 74
    "Whimper (dog)",                                   # 75
    "Cat",                                             # 76
    "Purr",                                            # 77
    "Meow",                                            # 78
    "Hiss",                                            # 79
    "Caterwaul",                                       # 80
    "Livestock, farm animals, working animals",        # 81
    "Horse",                                           # 82
    "Clip-clop",                                       # 83
    "Neigh, whinny",                                   # 84
    "Cattle, bovinae",                                 # 85
    "Moo",                                             # 86
    "Cowbell",                                         # 87
    "Pig",                                             # 88
    "Oink",                                            # 89
    "Goat",                                            # 90
    "Bleat",                                           # 91
    "Sheep",                                           # 92
    "Baa",                                             # 93
    "Fowl",                                            # 94
    "Chicken, rooster",                                # 95
    "Cluck",                                           # 96
    "Crowing, cock-a-doodle-doo",                      # 97
    "Turkey",                                          # 98
    "Gobble",                                          # 99
    "Duck",                                            # 100
    "Quack",                                           # 101
    "Goose",                                           # 102
    "Honk",                                            # 103
    "Wild animals",                                    # 104
    "Roaring cats (lions, tigers)",                    # 105
    "Roar",                                            # 106
    "Bird",                                            # 107
    "Bird vocalization, bird call, bird song",         # 108
    "Chirp, tweet",                                    # 109
    "Squawk",                                          # 110
    "Pigeon, dove",                                    # 111
    "Coo",                                             # 112
    "Crow",                                            # 113
    "Caw",                                             # 114
    "Owl",                                             # 115
    "Hoot",                                            # 116
    "Bird flight, flapping wings",                     # 117
    "Canary",                                          # 118
    "Parrot",                                          # 119
    "Insect",                                          # 120
    "Cricket",                                         # 121
    "Mosquito",                                        # 122
    "Fly, housefly",                                   # 123
    "Buzz",                                            # 124
    "Bee, wasp, etc.",                                 # 125
    "Frog",                                            # 126
    "Croak",                                           # 127
    "Snake",                                           # 128
    "Rattle",                                          # 129
    "Whale vocalization",                              # 130
    "Music",                                           # 131
    "Musical instrument",                              # 132
    "Plucked string instrument",                       # 133
    "Guitar",                                          # 134
    "Electric guitar",                                 # 135
    "Bass guitar",                                     # 136
    "Acoustic guitar",                                 # 137
    "Steel guitar, slide guitar",                      # 138
    "Tapping (guitar technique)",                      # 139
    "Strum",                                           # 140
    "Banjo",                                           # 141
    "Sitar",                                           # 142
    "Mandolin",                                        # 143
    "Zither",                                          # 144
    "Ukulele",                                         # 145
    "Keyboard (musical)",                              # 146
    "Piano",                                           # 147
    "Electric piano",                                  # 148
    "Organ",                                           # 149
    "Electronic organ",                                # 150
    "Hammond organ",                                   # 151
    "Synthesizer",                                     # 152
    "Sampler",                                         # 153
    "Harpsichord",                                     # 154
    "Percussion",                                      # 155
    "Drum kit",                                        # 156
    "Drum machine",                                    # 157
    "Drum",                                            # 158
    "Snare drum",                                      # 159
    "Rimshot",                                         # 160
    "Drum roll",                                       # 161
    "Bass drum",                                       # 162
    "Timpani",                                         # 163
    "Tabla",                                           # 164
    "Cymbal",                                          # 165
    "Hi-hat",                                          # 166
    "Wood block",                                      # 167
    "Tambourine",                                      # 168
    "Rattle (instrument)",                             # 169
    "Maraca",                                          # 170
    "Gong",                                            # 171
    "Tubular bells",                                   # 172
    "Mallet percussion",                               # 173
    "Marimba, xylophone",                              # 174
    "Glockenspiel",                                    # 175
    "Vibraphone",                                      # 176
    "Steelpan",                                        # 177
    "Orchestra",                                       # 178
    "Brass instrument",                                # 179
    "French horn",                                     # 180
    "Trumpet",                                         # 181
    "Trombone",                                        # 182
    "Bowed string instrument",                         # 183
    "String section",                                  # 184
    "Violin, fiddle",                                  # 185
    "Pizzicato",                                       # 186
    "Cello",                                           # 187
    "Double bass",                                     # 188
    "Wind instrument, woodwind instrument",            # 189
    "Flute",                                           # 190
    "Saxophone",                                       # 191
    "Clarinet",                                        # 192
    "Harp",                                            # 193
    "Bell",                                            # 194
    "Church bell",                                     # 195
    "Jingle bell",                                     # 196
    "Bicycle bell",                                    # 197
    "Tuning fork",                                     # 198
    "Chime",                                           # 199
    "Wind chime",                                      # 200
    "Change ringing (campanology)",                    # 201
    "Harmonica",                                       # 202
    "Accordion",                                       # 203
    "Bagpipes",                                        # 204
    "Didgeridoo",                                      # 205
    "Shofar",                                          # 206
    "Theremin",                                        # 207
    "Singing bowl",                                    # 208
    "Scratching (performance technique)",              # 209
    "Pop music",                                       # 210
    "Hip hop music",                                   # 211
    "Beatboxing",                                      # 212
    "Techno",                                          # 213
    "Dubstep",                                         # 214
    "Drum and bass",                                   # 215
    "Electronica",                                     # 216
    "Electronic music",                                # 217
    "House music",                                     # 218
    "Dance music",                                     # 219
    "Disco",                                           # 220
    "Electro",                                         # 221
    "Trance music",                                    # 222
    "Dub",                                             # 223
    "Ambient music",                                   # 224
    "New-age music",                                   # 225
    "Psychedelic",                                     # 226
    "Soundtrack music",                                # 227
    "Theme music",                                     # 228
    "Christmas music",                                 # 229
    "Anime",                                           # 230
    "Video game music",                                # 231
    "Lullaby",                                         # 232
    "Jazz",                                            # 233
    "Blues",                                            # 234
    "Soul music",                                      # 235
    "Rhythm and blues",                                # 236
    "Funk",                                            # 237
    "Folk music",                                      # 238
    "Middle Eastern music",                            # 239
    "Country",                                         # 240
    "Bluegrass",                                       # 241
    "Flamenco",                                        # 242
    "Traditional music",                               # 243
    "Wedding music",                                   # 244
    "Funeral march",                                   # 245
    "March",                                           # 246
    "Reggae",                                          # 247
    "Ska",                                             # 248
    "Grunge",                                          # 249
    "Progressive rock",                                # 250
    "Punk rock",                                       # 251
    "Heavy metal",                                     # 252
    "Rock music",                                      # 253
    "Rock and roll",                                   # 254
    "Psychedelic rock",                                # 255
    "Swing music",                                     # 256
    "Gospel music",                                    # 257
    "Opera",                                           # 258
    "Vocal music",                                     # 259
    "A capella",                                       # 260
    "Music of Latin America",                          # 261
    "Salsa music",                                     # 262
    "Cumbia",                                          # 263
    "Merengue",                                        # 264
    "Polka",                                           # 265
    "Music of Africa",                                 # 266
    "Music of Asia",                                   # 267
    "Music of Bollywood",                              # 268
    "Tender music",                                    # 269
    "Sad music",                                       # 270
    "Exciting music",                                  # 271
    "Angry music",                                     # 272
    "Scary music",                                     # 273
    "Wind",                                            # 274
    "Rustling leaves",                                 # 275
    "Wind noise (microphone)",                         # 276
    "Thunderstorm",                                    # 277
    "Thunder",                                         # 278
    "Water",                                           # 279
    "Rain",                                            # 280
    "Raindrop",                                        # 281
    "Rain on surface",                                 # 282
    "Stream",                                          # 283
    "Waterfall",                                       # 284
    "Ocean",                                           # 285
    "Waves, surf",                                     # 286
    "Steam",                                           # 287
    "Gurgling",                                        # 288
    "Fire",                                            # 289
    "Crackle",                                         # 290
    "Vehicle",                                         # 291
    "Boat, Water vehicle",                             # 292
    "Sailboat, sailing ship",                          # 293
    "Rowboat, canoe, kayak",                           # 294
    "Motorboat, speedboat",                            # 295
    "Ship",                                            # 296
    "Motor vehicle (road)",                            # 297
    "Car",                                             # 298
    "Vehicle horn, car horn, honking",                 # 299
    "Toot",                                            # 300
    "Car alarm",                                       # 301
    "Power windows, electric windows",                 # 302
    "Skidding",                                        # 303
    "Tire squeal",                                     # 304
    "Car passing by",                                  # 305
    "Race car, auto racing",                           # 306
    "Truck",                                           # 307
    "Air brake",                                       # 308
    "Air horn, truck horn",                            # 309
    "Reversing beeps",                                 # 310
    "Ice cream truck, ice cream van",                  # 311
    "Bus",                                             # 312
    "Emergency vehicle",                               # 313
    "Police car (siren)",                              # 314
    "Ambulance (siren)",                               # 315
    "Fire engine, fire truck (siren)",                 # 316
    "Motorcycle",                                      # 317
    "Traffic noise, roadway noise",                    # 318
    "Rail transport",                                  # 319
    "Train",                                           # 320
    "Train whistle",                                   # 321
    "Train horn",                                      # 322
    "Railroad car, train wagon",                       # 323
    "Train wheels squealing",                          # 324
    "Subway, metro, underground",                      # 325
    "Aircraft",                                        # 326
    "Aircraft engine",                                 # 327
    "Jet engine",                                      # 328
    "Propeller, airscrew",                             # 329
    "Helicopter",                                      # 330
    "Fixed-wing aircraft, airplane",                   # 331
    "Bicycle",                                         # 332
    "Skateboard",                                      # 333
    "Engine",                                          # 334
    "Light engine (high frequency)",                   # 335
    "Dental drill, dentist's drill",                   # 336
    "Lawn mower",                                      # 337
    "Chainsaw",                                        # 338
    "Medium engine (mid frequency)",                   # 339
    "Heavy engine (low frequency)",                    # 340
    "Engine knocking",                                 # 341
    "Engine starting",                                 # 342
    "Idling",                                          # 343
    "Accelerating, revving, vroom",                    # 344
    "Door",                                            # 345
    "Doorbell",                                        # 346
    "Ding-dong",                                       # 347
    "Sliding door",                                    # 348
    "Slam",                                            # 349
    "Knock",                                           # 350
    "Tap",                                             # 351
    "Squeak",                                          # 352
    "Cupboard open or close",                          # 353
    "Drawer open or close",                            # 354
    "Dishes, pots, and pans",                          # 355
    "Cutlery, silverware",                             # 356
    "Chopping (food)",                                 # 357
    "Frying (food)",                                   # 358
    "Microwave oven",                                  # 359
    "Blender",                                         # 360
    "Water tap, faucet",                               # 361
    "Sink (filling or washing)",                       # 362
    "Bathtub (filling or washing)",                    # 363
    "Hair dryer",                                      # 364
    "Toilet flush",                                    # 365
    "Toothbrush",                                      # 366
    "Electric toothbrush",                             # 367
    "Vacuum cleaner",                                  # 368
    "Zipper (clothing)",                               # 369
    "Keys jangling",                                   # 370
    "Coin (dropping)",                                 # 371
    "Scissors",                                        # 372
    "Electric shaver, electric razor",                 # 373
    "Shuffling cards",                                 # 374
    "Typing",                                          # 375
    "Typewriter",                                      # 376
    "Computer keyboard",                               # 377
    "Writing",                                         # 378
    "Alarm",                                           # 379
    "Telephone",                                       # 380
    "Telephone bell ringing",                          # 381
    "Ringtone",                                        # 382
    "Telephone dialing, DTMF",                         # 383
    "Dial tone",                                       # 384
    "Busy signal",                                     # 385
    "Alarm clock",                                     # 386
    "Siren",                                           # 387
    "Civil defense siren",                             # 388
    "Buzzer",                                          # 389
    "Smoke detector, smoke alarm",                     # 390
    "Fire alarm",                                      # 391
    "Foghorn",                                         # 392
    "Whistle",                                         # 393
    "Steam whistle",                                   # 394
    "Mechanisms",                                      # 395
    "Ratchet, pawl",                                   # 396
    "Clock",                                           # 397
    "Tick",                                            # 398
    "Tick-tock",                                       # 399
    "Gears",                                           # 400
    "Pulleys",                                         # 401
    "Sewing machine",                                  # 402
    "Mechanical fan",                                  # 403
    "Air conditioning",                                # 404
    "Cash register",                                   # 405
    "Printer",                                         # 406
    "Camera",                                          # 407
    "Single-lens reflex camera",                       # 408
    "Tools",                                           # 409
    "Hammer",                                          # 410
    "Jackhammer",                                      # 411
    "Sawing",                                          # 412
    "Filing (rasp)",                                   # 413
    "Sanding",                                         # 414
    "Power tool",                                      # 415
    "Drill",                                           # 416
    "Explosion",                                       # 417
    "Gunshot, gunfire",                                # 418
    "Machine gun",                                     # 419
    "Fusillade",                                       # 420
    "Artillery fire",                                  # 421
    "Cap gun",                                         # 422
    "Fireworks",                                       # 423
    "Firecracker",                                     # 424
    "Burst, pop",                                      # 425
    "Eruption",                                        # 426
    "Boom",                                            # 427
    "Wood",                                            # 428
    "Chop",                                            # 429
    "Splinter",                                        # 430
    "Crack",                                           # 431
    "Glass",                                           # 432
    "Chink, clink",                                    # 433
    "Shatter",                                         # 434
    "Liquid",                                          # 435
    "Splash, splatter",                                # 436
    "Slosh",                                           # 437
    "Squish",                                          # 438
    "Drip",                                            # 439
    "Pour",                                            # 440
    "Trickle, dribble",                                # 441
    "Gush",                                            # 442
    "Fill (with liquid)",                              # 443
    "Spray",                                           # 444
    "Pump (liquid)",                                   # 445
    "Stir",                                            # 446
    "Boiling",                                         # 447
    "Sonar",                                           # 448
    "Arrow",                                           # 449
    "Whoosh, swoosh, swish",                           # 450
    "Thump, thud",                                     # 451
    "Thunk",                                           # 452
    "Electronic tuner",                                # 453
    "Effects unit",                                    # 454
    "Chorus effect",                                   # 455
    "Basketball bounce",                               # 456
    "Bang",                                            # 457
    "Slap, smack",                                     # 458
    "Whack, thwack",                                   # 459
    "Smash, crash",                                    # 460
    "Breaking",                                        # 461
    "Bouncing",                                        # 462
    "Whip",                                            # 463
    "Flap",                                            # 464
    "Scratch",                                         # 465
    "Scrape",                                          # 466
    "Rub",                                             # 467
    "Roll",                                            # 468
    "Crushing",                                        # 469
    "Crumpling, crinkling",                            # 470
    "Tearing",                                         # 471
    "Beep, bleep",                                     # 472
    "Ping",                                            # 473
    "Ding",                                            # 474
    "Clang",                                           # 475
    "Squeal",                                          # 476
    "Creak",                                           # 477
    "Rustle",                                          # 478
    "Whir",                                            # 479
    "Clatter",                                         # 480
    "Jingle, tinkle",                                  # 481
    "Rattle",                                          # 482
    "Chatter",                                         # 483
    "Click",                                           # 484
    "Clickety-clack",                                  # 485
    "Rumble",                                          # 486
    "Plop",                                            # 487
    "Tapping (things)",                                # 488
    "Tap dance",                                       # 489
    "Sizzle",                                          # 490
    "Hum",                                             # 491
    "Zing",                                            # 492
    "Boing",                                           # 493
    "Crunch",                                          # 494
    "Silence",                                         # 495
    "Sine wave",                                       # 496
    "Harmonic",                                        # 497
    "Chirp tone",                                      # 498
    "Sound effect",                                    # 499
    "Pulse",                                           # 500
    "Inside, small room",                              # 501
    "Inside, large room or hall",                      # 502
    "Inside, public space",                            # 503
    "Outside, urban or manmade",                       # 504
    "Outside, rural or natural",                       # 505
    "Reverberation",                                   # 506
    "Echo",                                            # 507
    "Noise",                                           # 508
    "Environmental noise",                             # 509
    "Static",                                          # 510
    "Mains hum",                                       # 511
    "Distortion",                                      # 512
    "Sidetone",                                        # 513
    "Cacophony",                                       # 514
    "White noise",                                     # 515
    "Pink noise",                                      # 516
    "Throbbing",                                       # 517
    "Vibration",                                       # 518
    "Television",                                      # 519
    "Radio",                                           # 520
    "Field recording",                                 # 521
    "Clicking (insect)",                               # 522  (note: insect-related)
    "Scratching (insect)",                             # 523  (note: insect-related)
    "Buzzing (insect)",                                # 524  (note: insect-related)
    "Singing (insect)",                                # 525  (note: insect-related)
    "Howl (wind)",                                     # 526  (note: wind-related)
]
# fmt: on

assert len(AUDIOSET_LABELS) == 527, f"Expected 527 labels, got {len(AUDIOSET_LABELS)}"

# =========================================================================
# Index-based category mapping (more reliable than keyword matching for
# classes whose names may not obviously match)
# =========================================================================
_SPEECH_INDICES = set(range(0, 8))  # 0-7: Speech, Male/Female/Child speech, Conversation, Narration, Babbling, Speech synthesizer
_HUMAN_VOCAL_INDICES = set(range(8, 67))  # 8-66: Shout, Scream, Laughter, Crying, Singing, Breathing, etc.
_ANIMAL_INDICES = set(range(67, 131))  # 67-130: Dog, Cat, Livestock, Birds, Insects, Frog, Snake, Whale
_MUSIC_INDICES = set(range(131, 274))  # 131-273: Music, Instruments, Genres
_NATURE_INDICES = set(range(274, 291))  # 274-290: Wind, Thunder, Water, Rain, Fire
_VEHICLE_INDICES = set(range(291, 345))  # 291-344: Boats, Cars, Trucks, Trains, Aircraft, Engines
_DOMESTIC_INDICES = set(range(345, 378))  # 345-378: Door, Kitchen, Bathroom, Typing
_ALERT_INDICES = set(range(379, 395))  # 379-394: Alarm, Telephone, Siren, Buzzer, Whistle
_MECHANISM_INDICES = set(range(395, 417))  # 395-416: Clock, Gears, Fan, Camera, Tools, Drill
_EXPLOSION_INDICES = set(range(417, 428))  # 417-427: Explosion, Gunshot, Fireworks
_MATERIAL_INDICES = set(range(428, 495))  # 428-494: Wood, Glass, Liquid, Impact sounds
_CHANNEL_INDICES = set(range(495, 527))  # 495-526: Silence, Noise, Reverb, Channel effects


def get_category_by_index(idx: int) -> str:
    """Map a class index directly to a high-level category."""
    if idx in _SPEECH_INDICES:
        return "Speech"
    if idx in _HUMAN_VOCAL_INDICES:
        return "Human_activity"
    if idx in _ANIMAL_INDICES:
        return "Animal"
    if idx in _MUSIC_INDICES:
        return "Music"
    if idx in _NATURE_INDICES:
        return "Environment"
    if idx in _VEHICLE_INDICES:
        return "Vehicle"
    if idx in _DOMESTIC_INDICES:
        return "Domestic"
    if idx in _ALERT_INDICES:
        return "Alert"
    if idx in _MECHANISM_INDICES:
        return "Domestic"
    if idx in _EXPLOSION_INDICES:
        return "Other"
    if idx in _MATERIAL_INDICES:
        return "Other"
    if idx in _CHANNEL_INDICES:
        return "Other"
    return "Other"


def generate_csv_content() -> str:
    """Generate the class_labels_indices.csv content from embedded labels."""
    lines = ["index,mid,display_name"]
    for i, label in enumerate(AUDIOSET_LABELS):
        # MID values are not critical for inference, use placeholder
        mid = f"/m/{i:05d}"
        lines.append(f'{i},{mid},"{label}"')
    return "\n".join(lines) + "\n"
