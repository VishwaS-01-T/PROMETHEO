# Campaign-AI

Campaign-AI is an AI-powered platform designed to streamline campaign management, content creation, and research workflows. It combines a Python backend with a modern React frontend, enabling users to generate, control, and analyze campaign content efficiently.

## Features
- AI-driven prompt generation and content creation
- Campaign breakdown and reporting tools
- Research and web editing capabilities
- Interactive and animated UI components
- Dark mode support

## Project Structure

```
Campaign-AI/
├── foundry_server.py         # Python backend server
├── prompt.py                # AI prompt logic
├── sch.py                   # Scheduling or schema logic
├── requirements.txt         # Python dependencies
├── run.sh                   # Shell script to run backend
├── package.json             # Node.js dependencies (root)
├── README.md                # Project documentation
└── ai-foundry-frontend/     # React frontend
	 ├── index.html
	 ├── package.json         # Frontend dependencies
	 ├── vite.config.js       # Vite configuration
	 ├── public/              # Static assets
	 ├── src/                 # Source code
	 │   ├── App.jsx          # Main React app
	 │   ├── main.jsx         # Entry point
	 │   ├── styles.css       # Global styles
	 │   ├── components/      # UI components
	 │   ├── hooks/           # Custom hooks
	 │   ├── pages/           # Page components
	 │   └── ...
```

## Getting Started

### Backend Setup
1. Install Python dependencies:
	```bash
	pip install -r requirements.txt
	```
2. Run the backend server:
	```bash
	./run.sh
	```

### Frontend Setup
1. Navigate to the frontend directory:
	```bash
	cd ai-foundry-frontend
	```
2. Install Node.js dependencies:
	```bash
	npm install
	```
3. Start the frontend development server:
	```bash
	npm run dev
	```

## Usage
1. Start both backend and frontend servers.
2. Access the web interface at [http://localhost:5173](http://localhost:5173) (default Vite port).
3. Use the platform to generate campaign content, analyze breakdowns, and manage research.

## Contributing
Pull requests and suggestions are welcome. Please open an issue to discuss changes or improvements.

## License
This project is licensed under the MIT License.
3c51e723-0406-430e-a094-459c840117ce3c51e723-0406-430e-a094-459c840117ce