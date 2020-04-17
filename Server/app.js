const path = require("path");
// const spawn = require("child_process").spawn;

const express = require("express");
const bodyParser = require("body-parser");

const homepageRoutes = require("./routes/homepage");

const app = express();

app.use(bodyParser.urlencoded({ extended: false }));
app.use(express.static("public"));

// Home page
app.use(homepageRoutes);

// invalid page
app.use((req, res, next) => {
  res.status(404).sendFile(path.join(__dirname, "views", "404.html"));
});

app.listen(3000);

/* const pythonProcess = spawn("python", [
  path.join("python", "test.py"),
  5,
  12,
]);
pythonProcess.stdout.on("data", (data) => {
  res.send(`<h1>${data}</h1>`);
}); */
