const express = require("express");
const bodyParser = require("body-parser");
const request = require('request');
const path = require('path');
const axios = require('axios');

const app = express();

// Set up EJS view engine
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'templates'));

// Middleware
app.use(bodyParser.urlencoded({ extended: true }));

// Correctly serve static files with proper MIME types
// This serves files from the project root directory
app.use(express.static(path.join(__dirname, '..')));

// Also serve from the regular public directory for backward compatibility
app.use(express.static(path.join(__dirname, 'public')));

// Set proper content types for common file extensions
app.use((req, res, next) => {
  const ext = path.extname(req.url);
  if (ext === '.css') {
    res.type('text/css');
  } else if (ext === '.js') {
    res.type('application/javascript');
  }
  next();
});

// Game URLs
const url_endlessRunner_1 = "https://scratch.mit.edu/projects/600560206/embed";
const url_endlessRunner_2 = "https://scratch.mit.edu/projects/599708735/embed";
const url_endlessRunner_3 = "https://scratch.mit.edu/projects/600097774/embed";
let url_send = "";

// API endpoints for communicating with Flask backend
const FLASK_API = "http://localhost:5000";

// Routes
app.get("/", function (req, res) {
    res.render("index");
});

app.post("/", function (req, res) {
    console.log(req.body);
    const play = req.body.play;
    
    if (play === "yes") {
        res.redirect("/yoga");
    }
});

// Fixed yoga route - ensure it's only rendering once
app.get("/yoga", function (req, res) {
    // Make sure we're only rendering the template once
    // and not making duplicate calls or loading content twice
    res.render("yoga", {
        // Add any data you need to pass to the template
        pageTitle: "Yoga Session"
    });
});

// API endpoint to check yoga pose status from Flask backend
app.get("/api/yoga-status", async function (req, res) {
    try {
        const response = await axios.get(`${FLASK_API}/yoga/yoga-status`);
        res.json(response.data);
    } catch (error) {
        console.error("Error fetching yoga status:", error);
        res.status(500).json({ error: "Failed to fetch yoga status" });
    }
});

app.post("/api/start-yoga", async function (req, res) {
    try {
        const response = await axios.post(`${FLASK_API}/yoga/start-yoga`);
        res.json(response.data);
    } catch (error) {
        console.error("Error starting yoga session:", error);
        res.status(500).json({ error: "Failed to start yoga session" });
    }
});

app.post("/api/stop-yoga", async function (req, res) {
    try {
        const response = await axios.post(`${FLASK_API}/yoga/stop-yoga`);
        res.json(response.data);
    } catch (error) {
        console.error("Error stopping yoga session:", error);
        res.status(500).json({ error: "Failed to stop yoga session" });
    }
});

// Game routes - render the specific game pages
app.get("/play1", function (req, res) {
    res.render("play1", { 
        gameSrc: url_endlessRunner_1,
        gameTitle: "Game 1" 
    });
});

app.get("/play2", function (req, res) {
    res.render("play2", { 
        gameSrc: url_endlessRunner_2,
        gameTitle: "Game 2" 
    });
});

app.get("/play3", function (req, res) {
    res.render("play3", { 
        gameSrc: url_endlessRunner_3,
        gameTitle: "Game 3" 
    });
});

// Games home page
app.get("/games", function (req, res) {
    res.render("games");
});

// Start server
const PORT = process.env.PORT || 3000;
app.listen(PORT, function () {
    console.log(`Frontend server is running on port ${PORT}`);
    console.log(`Visit http://localhost:${PORT} to view the website`);
});