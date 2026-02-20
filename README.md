
# FPL Simple Squad Selector

This tiny utility consumes Fantasy Premier League's public bootstrap endpoint and picks a squad by multiplying each player's ICT Index by their total points. It then returns:

- 2 Goalkeepers
- 5 Defenders
- 5 Midfielders
- 3 Forwards

## Table of Contents
- [Setup](#setup)
- [Usage](#usage)
- [Notes](#notes)
- [File Overview](#file-overview)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

The script prints the best players at each position sorted by the ICT × total-points metric.

## Notes

- Data is fetched live from `https://fantasy.premierleague.com/api/bootstrap-static/`.
- Network access is required. If the request fails, rerun once connectivity is restored.

## File Overview

*Provide a brief explanation of key files if desired.*

## Contributing
We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md) for details.

## License
This project is licensed under the [MIT License](LICENSE).

## Contact
For questions or support, please open an issue or contact the maintainer via [X](https://x.com/elxecutor/).
