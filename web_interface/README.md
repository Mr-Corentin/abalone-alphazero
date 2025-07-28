# Abalone Web Interface

A web-based interface for playing Abalone using your AlphaZero implementation.

## Architecture

```
Frontend (React) ↔ Backend (FastAPI) ↔ AbaloneEnv (Your existing code)
```

## Setup Instructions

### 1. Backend Setup

```bash
# Navigate to backend directory
cd web_interface/backend

# Install Python dependencies
pip install -r requirements.txt

# Start the FastAPI server
python main.py
```

The backend server will start on `http://localhost:8000`

### 2. Frontend Setup

```bash
# Navigate to frontend directory
cd web_interface/frontend

# Install Node.js dependencies
npm install

# Start the React development server
npm start
```

The frontend will open in your browser at `http://localhost:3000`

## How to Play

1. **Start the game** - Backend initializes with Belgian Daisy setup
2. **Select marbles** - Click on your marbles (Black starts first)
3. **Choose direction** - Click direction arrows around selected marbles
4. **Move validation** - All moves are validated using your AbaloneEnv
5. **Game rules** - Full Abalone rules including pushes and captures

## Features

✅ **Complete Integration**
- Uses your existing `AbaloneEnv` for all game logic
- Handles canonical representation automatically
- Full legal move validation
- Multi-marble pushes and captures
- Game end detection

✅ **Interactive UI**
- Click-to-select marble interface
- Visual direction arrows
- Real-time game state updates
- Error handling and feedback

✅ **API Endpoints**
- `POST /game/new` - Start new game
- `GET /game/state` - Get current state
- `POST /game/move` - Execute move
- `GET /game/legal-moves` - Get legal moves
- `GET /debug/board` - Debug board display

## Development

- **Frontend hot reload** - Changes appear immediately
- **Backend auto-reload** - Use `uvicorn main:app --reload` for development
- **CORS enabled** - Frontend can communicate with backend
- **Error handling** - Proper error messages for invalid moves

## Next Steps

- [ ] Add AI opponent integration
- [ ] Move history and undo
- [ ] Game analysis features
- [ ] Save/load games
- [ ] Multiplayer support