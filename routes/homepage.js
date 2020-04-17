const path = require("path");

const express = require("express");

const rootDir = require("../utils/root_dir");

const router = express.Router();

router.get("/", (req, res, next) => {
  res.sendFile(path.join(rootDir, "views", "homepage.html"));
});

module.exports = router;
