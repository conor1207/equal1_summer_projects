About

This repository collects the projects I worked on during my summer internship at Equal1 June-July 2027, supervised by Andrii Sokolov.

The work covered two areas of silicon quantum computing research:

Simulation: modelling how the valley splitting of electrons in silicon quantum dots depends on device parameters, using the QTCAD simulator.
Measurement analysis: loading data from microchip tests and producing clear plots of parameter sweeps (e.g. gate voltage vs. magnetic field) for a presentation.

Each project has its own folder and its own detailed README.

Projects
Project	What it does	Tools
Valley splitting simulations	Simulates silicon quantum dots and studies how valley splitting changes with change in a random Si/SiGe heterostructure	Python, QTCAD, [NumPy, Matplotlib]
Measurement plotting	Turns .db measurement files into 2D colour maps and line cuts of parameter sweeps	Python, [QCoDeS / sqlite3, pandas, Matplotlib]


Key outcomes:
Built reusable plotting tools used for figures in a team presentation
Modelled a Si/SiGe heterostructure's valley splitting in QTCAD


Repository structure
equal1-summer-projects/
├── README.md              ← this file (overview)
├── .gitignore             ← keeps data files, licences and environments out of Git
├── valley_splitting/
│   └── README.md          ← simulation project details
└── data_analysis/
    └── README.md          ← plotting project details


Data and software notes
No confidential data is included. Measurement data and unpublished results belong to Equal1 and are not part of this repository.
QTCAD is commercial software and is not included. Running the simulations requires a licensed installation.
Setup instructions for each project are in its own README.

Skills demonstrated:
Python for scientific computing and data analysis
Semiconductor quantum device simulation (QTCAD)
Loading and processing experimental data from databases
Scientific visualisation for technical presentations
Git and GitHub
writing documentation
Presentation skills

Reflection:

I learned the technical nature of research work. Working in quantum computing, an extremely technical subject itself I found myself extremely challenged and out of my depth with there being some crossover with my course and studies I have done but I had never seen anything in the depth required for industry. I was forced to learn quickly and adapt, teaching myself lots about valley splitting through research papers given to me by my supervisor. I also had little time to familiarise myself with the QTCAD software. It was an incredible learning experience and my only wish is that I had more time there to learn more about the subject and be able to contribute more to the project.

Acknowledgements

Thanks to Andrii sokolov and the Quantum Architecture team at Equal1 for their guidance and support during the project.