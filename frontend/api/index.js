const express = require("express");
const serverless = require("serverless-http");
const bodyParser = require("body-parser");
const path = require("path");

const app = express();

// EJS Setup
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, '../templates'));

// Middleware
app.use(bodyParser.urlencoded({ extended: true }));
app.use(express.static(path.join(__dirname, '../public')));

// Routes
app.get("/", (req, res) => {
  res.render("index");
});

app.get("/yoga", (req, res) => {
  res.render("yoga", { pageTitle: "Yoga Session" });
});

app.get("/games", (req, res) => {
  res.render("games");
});

app.get("/play1", (req, res) => {
  res.render("play1", { 
    gameSrc: "https://scratch.mit.edu/projects/600560206/embed",
    gameTitle: "Game 1"
  });
});

app.get("/play2", (req, res) => {
  res.render("play2", { 
    gameSrc: "https://scratch.mit.edu/projects/599708735/embed",
    gameTitle: "Game 2"
  });
});

app.get("/play3", (req, res) => {
  res.render("play3", { 
    gameSrc: "https://scratch.mit.edu/projects/600097774/embed",
    gameTitle: "Game 3"
  });
});

module.exports = serverless(app);
