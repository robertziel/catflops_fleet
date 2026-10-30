'use strict';
/* THE SHIPPED MODEL LIBRARY AS ROWS, for the panel's node tests: library/models.json merged by id with its recipe's
 * launch (and the measured tuned input / overhead from library/measurements.json) — the row shape the retired
 * catalog.json had, minus the deployment's folders (vllm_lb/catalog.py `document` is the LB's own assembler; the page
 * only ever sees its view in /lb/status). */
const fs = require('fs');
const path = require('path');

const LIBRARY = path.join(__dirname, '..', 'vllm_lb', 'library');
const read = (name) => JSON.parse(fs.readFileSync(path.join(LIBRARY, name), 'utf8'));

module.exports = function libraryRows() {
  const recipes = Object.fromEntries(read('recipes.json').recipes.map((r) => [r.model, r]));
  const measured = Object.fromEntries(read('measurements.json').measurements.map((r) => [r.model, r]));
  return read('models.json').models.map((m) => {
    const recipe = recipes[m.id] || { launch: {} };
    const meas = measured[m.id] || {};
    const launch = {};
    for (const [mode, p] of Object.entries(recipe.launch)) launch[mode] = { ...p, ...((meas.launch || {})[mode] || {}) };
    return { ...m, launch, node_engine: recipe.node_engine, memory: meas.memory, engine_memory: meas.engine_memory };
  });
};
