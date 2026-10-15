/**
 * NPH ML/Neuralnet Predictor - Data Engine
 * Handles CSV parsing, variable role & type deduction, missing value detection,
 * multi-strategy data imputation, feature encoding, scaling, and train/test splitting.
 * 
 * Conceived, designed and developed by Dr G Narenthiran MB ChB BSc(MedSci)(Hons) FEBNS FRCS(SN)
 */

(function (window) {
    'use strict';

    const DataEngine = {
        /**
         * RFC 4180 compliant CSV Parser with auto delimiter detection
         */
        parseCSV(csvText) {
            if (!csvText || typeof csvText !== 'string') {
                throw new Error("Empty or invalid CSV file content.");
            }

            // Detect delimiter: check first 3 lines
            const sampleLines = csvText.split(/\r\n|\n|\r/).filter(l => l.trim().length > 0).slice(0, 3);
            let delimiter = ',';
            if (sampleLines.length > 0) {
                const commaCount = (sampleLines[0].match(/,/g) || []).length;
                const semiCount = (sampleLines[0].match(/;/g) || []).length;
                const tabCount = (sampleLines[0].match(/\t/g) || []).length;
                if (semiCount > commaCount && semiCount >= tabCount) delimiter = ';';
                else if (tabCount > commaCount && tabCount >= semiCount) delimiter = '\t';
            }

            const rows = [];
            let currentRow = [];
            let currentField = '';
            let inQuotes = false;
            let i = 0;
            const len = csvText.length;

            while (i < len) {
                const char = csvText[i];
                const nextChar = csvText[i + 1];

                if (inQuotes) {
                    if (char === '"') {
                        if (nextChar === '"') {
                            currentField += '"';
                            i += 2;
                            continue;
                        } else {
                            inQuotes = false;
                            i++;
                            continue;
                        }
                    } else {
                        currentField += char;
                        i++;
                        continue;
                    }
                } else {
                    if (char === '"') {
                        inQuotes = true;
                        i++;
                        continue;
                    } else if (char === delimiter) {
                        currentRow.push(currentField.trim());
                        currentField = '';
                        i++;
                        continue;
                    } else if (char === '\r' || char === '\n') {
                        currentRow.push(currentField.trim());
                        currentField = '';
                        if (char === '\r' && nextChar === '\n') {
                            i += 2;
                        } else {
                            i++;
                        }
                        if (currentRow.some(val => val.length > 0)) {
                            rows.push(currentRow);
                        }
                        currentRow = [];
                        continue;
                    } else {
                        currentField += char;
                        i++;
                        continue;
                    }
                }
            }

            if (currentField.length > 0 || currentRow.length > 0) {
                currentRow.push(currentField.trim());
                if (currentRow.some(val => val.length > 0)) {
                    rows.push(currentRow);
                }
            }

            if (rows.length < 2) {
                throw new Error("CSV must contain at least a header row and one data row.");
            }

            const rawHeaders = rows[0];
            // Deduplicate headers if any
            const seenHeaders = {};
            const headers = rawHeaders.map((h, idx) => {
                let name = (h || `Column_${idx + 1}`).trim();
                if (seenHeaders[name]) {
                    seenHeaders[name]++;
                    name = `${name}_${seenHeaders[name]}`;
                } else {
                    seenHeaders[name] = 1;
                }
                return name;
            });

            const dataRows = [];
            for (let r = 1; r < rows.length; r++) {
                const row = rows[r];
                const obj = {};
                for (let c = 0; c < headers.length; c++) {
                    const rawVal = c < row.length ? row[c] : '';
                    obj[headers[c]] = rawVal;
                }
                dataRows.push(obj);
            }

            return {
                headers,
                rows: dataRows,
                delimiter,
                totalRows: dataRows.length,
                totalCols: headers.length
            };
        },

        /**
         * Checks if a cell value is missing / blank / NaN / NA
         */
        isMissing(val) {
            if (val === null || val === undefined) return true;
            const str = String(val).trim();
            if (str === '') return true;
            const lower = str.toLowerCase();
            return lower === 'na' || lower === 'nan' || lower === 'null' || lower === 'none' || lower === '?' || lower === '.' || lower === 'n/a';
        },

        /**
         * Analyzes dataset columns: missing counts, data types, distinct values, and heuristics
         */
        analyzeColumns(headers, rows) {
            const analysis = {};

            headers.forEach((col, colIdx) => {
                let missingCount = 0;
                const distinctMap = new Map();
                let numericCount = 0;
                let validCount = 0;
                const validNumbers = [];

                rows.forEach((row, rowIdx) => {
                    const val = row[col];
                    if (this.isMissing(val)) {
                        missingCount++;
                    } else {
                        validCount++;
                        const strVal = String(val).trim();
                        distinctMap.set(strVal, (distinctMap.get(strVal) || 0) + 1);

                        const num = Number(strVal);
                        if (!isNaN(num) && strVal !== '') {
                            numericCount++;
                            validNumbers.push(num);
                        }
                    }
                });

                const isNumeric = validCount > 0 && numericCount === validCount;
                const distinctCount = distinctMap.size;
                const distinctValues = Array.from(distinctMap.keys());

                // Heuristic Role determination
                let role = 'feature';
                const lowerCol = col.toLowerCase();

                if (lowerCol.match(/^(patient_id|id|mrn|record_id|patient|subjid|subject|hash)$/i) || (colIdx === 0 && distinctCount === validCount && !isNumeric)) {
                    role = 'meta';
                } else if (colIdx === headers.length - 1 || lowerCol.match(/^(target|outcome|response|shunt_response|label|class|diagnos|result|improvement)/i)) {
                    role = 'target';
                }

                // Heuristic Type determination
                let type = 'nominal';
                if (isNumeric) {
                    const allIntegers = validNumbers.every(n => Number.isInteger(n));
                    if (distinctCount <= 7 && allIntegers && distinctCount >= 2) {
                        type = 'ordinal';
                    } else if (distinctCount > 7 || !allIntegers) {
                        type = 'continuous';
                    } else {
                        type = 'nominal';
                    }
                } else {
                    // Check if string labels resemble ordinal grades (e.g. grade 1, grade 2 or mild, moderate, severe)
                    const lowerValues = distinctValues.map(v => v.toLowerCase());
                    const looksOrdinal = lowerValues.some(v => v.includes('mild') || v.includes('mod') || v.includes('sev') || v.includes('stage') || v.includes('grade') || v.match(/^[0-9]+$/));
                    if (looksOrdinal && distinctCount <= 7) {
                        type = 'ordinal';
                    } else {
                        type = 'nominal';
                    }
                }

                // Calculate summary stats for numbers
                let min = null, max = null, mean = null, median = null, mode = null;
                if (validNumbers.length > 0) {
                    validNumbers.sort((a, b) => a - b);
                    min = validNumbers[0];
                    max = validNumbers[validNumbers.length - 1];
                    const sum = validNumbers.reduce((acc, v) => acc + v, 0);
                    mean = Number((sum / validNumbers.length).toFixed(4));
                    const mid = Math.floor(validNumbers.length / 2);
                    median = validNumbers.length % 2 === 0 ? Number(((validNumbers[mid - 1] + validNumbers[mid]) / 2).toFixed(4)) : validNumbers[mid];
                }

                // Calculate mode for any type
                let bestFreq = 0;
                distinctMap.forEach((freq, val) => {
                    if (freq > bestFreq) {
                        bestFreq = freq;
                        mode = isNumeric ? Number(val) : val;
                    }
                });

                analysis[col] = {
                    name: col,
                    index: colIdx,
                    total: rows.length,
                    validCount,
                    missingCount,
                    missingPercent: Number(((missingCount / rows.length) * 100).toFixed(1)),
                    distinctCount,
                    distinctValues,
                    isNumeric,
                    role, // 'meta', 'target', 'feature'
                    type, // 'continuous', 'ordinal', 'nominal'
                    stats: {
                        min,
                        max,
                        mean,
                        median,
                        mode,
                        distribution: Object.fromEntries(distinctMap)
                    },
                    // Default imputation strategy suggestion
                    suggestedImputation: type === 'continuous' ? 'median' : 'mode'
                };
            });

            return analysis;
        },

        /**
         * Perform data imputation across columns based on user configuration
         * @param {Array} rows - array of row objects
         * @param {Object} columnConfigs - { [colName]: { role, type, imputationStrategy, customImputeVal } }
         * @param {Object} columnAnalysis - analysis output from analyzeColumns
         */
        imputeDataset(rows, columnConfigs, columnAnalysis) {
            let workingRows = rows.map(r => ({ ...r }));
            const audit = [];
            const rowsToDrop = new Set();

            // First pass: identify rows that must be dropped if user chose 'drop' on any column, or if target is missing
            Object.keys(columnConfigs).forEach(col => {
                const config = columnConfigs[col];
                if (config.role === 'meta') return; // skip meta columns

                // Clinical rule: Outcome/target must NEVER be fabricated or imputed. Missing outcomes are dropped.
                if (config.role === 'target' || config.imputationStrategy === 'drop') {
                    workingRows.forEach((row, rIdx) => {
                        if (this.isMissing(row[col])) {
                            rowsToDrop.add(rIdx);
                        }
                    });
                }
            });

            if (rowsToDrop.size > 0) {
                workingRows = workingRows.filter((_, idx) => !rowsToDrop.has(idx));
                audit.push({
                    type: 'drop_rows',
                    description: `Dropped ${rowsToDrop.size} rows with missing values in columns marked for row deletion or missing clinical outcome.`
                });
            }

            // Second pass: Compute imputation values on the filtered rows
            Object.keys(columnConfigs).forEach(col => {
                const config = columnConfigs[col];
                const analysis = columnAnalysis[col];
                if (config.role === 'meta' || config.role === 'target') return; // Target is NEVER imputed!
                if (config.imputationStrategy === 'drop') return;

                const strategy = config.imputationStrategy;
                const colType = config.type;

                // Collect valid values from workingRows
                const validValues = [];
                const validNumbers = [];

                workingRows.forEach(r => {
                    const val = r[col];
                    if (!this.isMissing(val)) {
                        validValues.push(val);
                        const n = Number(val);
                        if (!isNaN(n)) validNumbers.push(n);
                    }
                });

                let fillValue = null;

                if (strategy === 'mean') {
                    if (validNumbers.length > 0) {
                        const sum = validNumbers.reduce((a, b) => a + b, 0);
                        fillValue = Number((sum / validNumbers.length).toFixed(4));
                    } else {
                        fillValue = 0;
                    }
                } else if (strategy === 'median') {
                    if (validNumbers.length > 0) {
                        const sorted = [...validNumbers].sort((a, b) => a - b);
                        const mid = Math.floor(sorted.length / 2);
                        fillValue = sorted.length % 2 === 0 ? Number(((sorted[mid - 1] + sorted[mid]) / 2).toFixed(4)) : sorted[mid];
                    } else {
                        fillValue = 0;
                    }
                } else if (strategy === 'mode') {
                    const freqMap = new Map();
                    validValues.forEach(v => freqMap.set(String(v), (freqMap.get(String(v)) || 0) + 1));
                    let topCount = 0;
                    let topVal = null;
                    freqMap.forEach((count, v) => {
                        if (count > topCount) {
                            topCount = count;
                            topVal = v;
                        }
                    });
                    if (colType === 'continuous' || colType === 'ordinal') {
                        fillValue = topVal !== null && !isNaN(Number(topVal)) ? Number(topVal) : 0;
                    } else {
                        fillValue = topVal !== null ? topVal : 'Unknown';
                    }
                } else if (strategy === 'constant') {
                    if (colType === 'continuous' || colType === 'ordinal') {
                        fillValue = Number(config.customImputeVal !== undefined && config.customImputeVal !== '' ? config.customImputeVal : 0);
                    } else {
                        fillValue = String(config.customImputeVal || 'Missing');
                    }
                } else if (strategy === 'knn') {
                    // KNN Imputation for continuous variables
                    // Will fill cell-by-cell in third pass
                    fillValue = 'knn_dynamic';
                }

                // Perform the imputation
                let imputedCount = 0;
                if (strategy === 'knn') {
                    // Precompute continuous reference columns
                    const numericCols = Object.keys(columnConfigs).filter(c => columnConfigs[c].type === 'continuous' && c !== col);
                    workingRows.forEach((row, rowIdx) => {
                        if (this.isMissing(row[col])) {
                            // Find k nearest rows that have non-missing col
                            const k = 3;
                            const distances = [];
                            workingRows.forEach((otherRow, otherIdx) => {
                                if (otherIdx === rowIdx || this.isMissing(otherRow[col])) return;
                                let distSq = 0;
                                let matchedFeatures = 0;
                                numericCols.forEach(nCol => {
                                    if (!this.isMissing(row[nCol]) && !this.isMissing(otherRow[nCol])) {
                                        const d = Number(row[nCol]) - Number(otherRow[nCol]);
                                        distSq += d * d;
                                        matchedFeatures++;
                                    }
                                });
                                if (matchedFeatures > 0) {
                                    distances.push({ val: Number(otherRow[col]), dist: Math.sqrt(distSq) });
                                }
                            });

                            if (distances.length > 0) {
                                distances.sort((a, b) => a.dist - b.dist);
                                const neighbors = distances.slice(0, Math.min(k, distances.length));
                                const avg = neighbors.reduce((acc, item) => acc + item.val, 0) / neighbors.length;
                                row[col] = Number(avg.toFixed(4));
                                imputedCount++;
                            } else {
                                // fallback to mean
                                const fallback = validNumbers.length > 0 ? (validNumbers.reduce((a, b) => a + b, 0) / validNumbers.length) : 0;
                                row[col] = Number(fallback.toFixed(4));
                                imputedCount++;
                            }
                        }
                    });
                } else {
                    workingRows.forEach(row => {
                        if (this.isMissing(row[col])) {
                            row[col] = fillValue;
                            imputedCount++;
                        } else if (colType === 'continuous' || colType === 'ordinal') {
                            const n = Number(row[col]);
                            if (!isNaN(n)) row[col] = n;
                        }
                    });
                }

                if (imputedCount > 0) {
                    audit.push({
                        type: 'impute_column',
                        column: col,
                        strategy,
                        imputedCount,
                        filledValue: fillValue === 'knn_dynamic' ? 'KNN Distance Weighted' : fillValue
                    });
                }
            });

            return {
                imputedRows: workingRows,
                audit,
                totalRowsAfter: workingRows.length
            };
        },

        /**
         * Preprocess features and target with optional explicit featureCols and targetCol
         */
        prepareDataset(rows, columnConfigs, featureCols = null, targetCol = null) {
            const configs = {};
            Object.keys(columnConfigs).forEach(c => {
                configs[c] = { ...columnConfigs[c] };
            });

            if (targetCol) {
                Object.keys(configs).forEach(c => {
                    if (configs[c].role === 'target') configs[c].role = 'meta';
                });
                if (configs[targetCol]) configs[targetCol].role = 'target';
                else configs[targetCol] = { role: 'target', type: 'nominal' };
            }

            if (featureCols && Array.isArray(featureCols)) {
                Object.keys(configs).forEach(c => {
                    if (c !== targetCol) configs[c].role = 'meta';
                });
                featureCols.forEach(c => {
                    if (configs[c]) configs[c].role = 'feature';
                    else configs[c] = { role: 'feature', type: 'continuous' };
                });
            }

            return this.prepareFeaturesAndTarget(rows, configs);
        },

        /**
         * Preprocess features and target into numeric arrays for machine learning models
         */
        prepareFeaturesAndTarget(rows, columnConfigs) {
            const featureCols = [];
            let targetCol = null;

            Object.keys(columnConfigs).forEach(col => {
                const conf = columnConfigs[col];
                if (conf.role === 'feature') {
                    featureCols.push(col);
                } else if (conf.role === 'target') {
                    targetCol = col;
                }
            });

            if (!targetCol) {
                throw new Error("Please select exactly one Target variable to predict.");
            }
            if (featureCols.length === 0) {
                throw new Error("Please select at least one Feature (predictor) variable.");
            }

            const targetConf = columnConfigs[targetCol];
            const isClassification = targetConf.type === 'nominal' || targetConf.type === 'ordinal';

            // 1. Build Target Encoder
            let targetClasses = [];
            let targetClassToIdx = {};
            let targetIdxToClass = {};
            let targetStats = null;

            if (isClassification) {
                // Collect distinct target classes
                const classSet = new Set();
                rows.forEach(r => {
                    const val = String(r[targetCol]);
                    classSet.add(val);
                });
                targetClasses = Array.from(classSet);
                // Sort classes naturally
                targetClasses.sort();
                targetClasses.forEach((cls, idx) => {
                    targetClassToIdx[cls] = idx;
                    targetIdxToClass[idx] = cls;
                });
            } else {
                // Regression target: collect numbers
                const nums = rows.map(r => Number(r[targetCol])).filter(n => !isNaN(n));
                const sum = nums.reduce((a, b) => a + b, 0);
                const mean = sum / nums.length;
                const variance = nums.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / nums.length;
                targetStats = {
                    mean,
                    std: Math.sqrt(variance) || 1,
                    min: Math.min(...nums),
                    max: Math.max(...nums)
                };
            }

            // 2. Build Feature Encoders
            // For Nominal: One-Hot Encoding map
            // For Ordinal: Natural / sorted rank mapping
            // For Continuous: Standardization (mean, std)
            const featureMetadata = {};
            let totalEncodedFeatures = 0;
            const encodedFeatureNames = [];

            featureCols.forEach(col => {
                const conf = columnConfigs[col];
                if (conf.type === 'nominal') {
                    const catSet = new Set();
                    rows.forEach(r => catSet.add(String(r[col])));
                    const categories = Array.from(catSet).sort();
                    featureMetadata[col] = {
                        type: 'nominal',
                        categories,
                        startIndex: totalEncodedFeatures,
                        length: categories.length
                    };
                    categories.forEach(cat => {
                        encodedFeatureNames.push(`${col}_${cat}`);
                    });
                    totalEncodedFeatures += categories.length;
                } else if (conf.type === 'ordinal') {
                    // Ordinal mapping: map distinct values to integers 0, 1, 2...
                    // Check if already numeric or string
                    const valSet = new Set();
                    rows.forEach(r => valSet.add(r[col]));
                    let ranks = Array.from(valSet);
                    const allNum = ranks.every(v => !isNaN(Number(v)));
                    if (allNum) {
                        ranks.sort((a, b) => Number(a) - Number(b));
                    } else {
                        // Intelligent clinical ordering for standard qualitative scales
                        const clinicalScales = [
                            ['none', 'mild', 'moderate', 'severe'],
                            ['absent', 'present'],
                            ['no', 'yes'],
                            ['low', 'medium', 'high', 'very high'],
                            ['stage 0', 'stage 1', 'stage 2', 'stage 3', 'stage 4'],
                            ['grade 0', 'grade 1', 'grade 2', 'grade 3', 'grade 4'],
                            ['grade i', 'grade ii', 'grade iii', 'grade iv']
                        ];
                        const lowerRanks = ranks.map(r => String(r).toLowerCase().trim());
                        let matchedScale = null;
                        for (const scale of clinicalScales) {
                            if (lowerRanks.every(r => scale.includes(r))) {
                                matchedScale = scale;
                                break;
                            }
                        }
                        if (matchedScale) {
                            ranks.sort((a, b) => {
                                const idxA = matchedScale.indexOf(String(a).toLowerCase().trim());
                                const idxB = matchedScale.indexOf(String(b).toLowerCase().trim());
                                return idxA - idxB;
                            });
                        } else {
                            ranks.sort();
                        }
                    }
                    const rankMap = {};
                    ranks.forEach((val, rk) => {
                        rankMap[String(val)] = rk;
                    });
                    featureMetadata[col] = {
                        type: 'ordinal',
                        ranks,
                        rankMap,
                        startIndex: totalEncodedFeatures,
                        length: 1
                    };
                    encodedFeatureNames.push(col);
                    totalEncodedFeatures += 1;
                } else {
                    // Continuous: compute mean and standard deviation
                    const nums = rows.map(r => Number(r[col])).filter(n => !isNaN(n));
                    const sum = nums.reduce((a, b) => a + b, 0);
                    const mean = sum / (nums.length || 1);
                    const variance = nums.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / (nums.length || 1);
                    const std = Math.sqrt(variance) || 1;
                    featureMetadata[col] = {
                        type: 'continuous',
                        mean,
                        std,
                        min: Math.min(...nums),
                        max: Math.max(...nums),
                        startIndex: totalEncodedFeatures,
                        length: 1
                    };
                    encodedFeatureNames.push(col);
                    totalEncodedFeatures += 1;
                }
            });

            // 3. Transform Rows into Matrix X and Vector y
            const X = [];
            const y = [];

            rows.forEach(row => {
                const xRow = new Array(totalEncodedFeatures).fill(0);
                featureCols.forEach(col => {
                    const meta = featureMetadata[col];
                    const val = row[col];

                    if (meta.type === 'nominal') {
                        const catIdx = meta.categories.indexOf(String(val));
                        if (catIdx !== -1) {
                            xRow[meta.startIndex + catIdx] = 1;
                        }
                    } else if (meta.type === 'ordinal') {
                        const rankVal = meta.rankMap[String(val)] !== undefined ? meta.rankMap[String(val)] : 0;
                        xRow[meta.startIndex] = rankVal;
                    } else {
                        // Continuous: standardized
                        const num = Number(val);
                        const scaled = (num - meta.mean) / meta.std;
                        xRow[meta.startIndex] = isNaN(scaled) ? 0 : Number(scaled.toFixed(5));
                    }
                });

                X.push(xRow);

                if (isClassification) {
                    const targetClass = String(row[targetCol]);
                    y.push(targetClassToIdx[targetClass] !== undefined ? targetClassToIdx[targetClass] : 0);
                } else {
                    y.push(Number(row[targetCol]));
                }
            });

            return {
                X,
                y,
                featureCols,
                targetCol,
                targetConf,
                isClassification,
                targetClasses,
                targetClassToIdx,
                targetIdxToClass,
                targetStats,
                featureMetadata,
                encodedFeatureNames,
                totalDimensions: totalEncodedFeatures
            };
        },

        /**
         * Transform a single arbitrary row (for single-case or batch inference)
         */
        transformSingleRow(inputRow, prepMeta) {
            const { featureCols, featureMetadata, totalDimensions } = prepMeta;
            const xRow = new Array(totalDimensions).fill(0);

            featureCols.forEach(col => {
                const meta = featureMetadata[col];
                const rawVal = inputRow[col];

                if (meta.type === 'nominal') {
                    const catIdx = meta.categories.indexOf(String(rawVal));
                    if (catIdx !== -1) {
                        xRow[meta.startIndex + catIdx] = 1;
                    }
                } else if (meta.type === 'ordinal') {
                    const rankVal = meta.rankMap[String(rawVal)] !== undefined ? meta.rankMap[String(rawVal)] : 0;
                    xRow[meta.startIndex] = rankVal;
                } else {
                    if (this.isMissing(rawVal) || rawVal === '' || rawVal === null || typeof rawVal === 'undefined' || isNaN(Number(rawVal))) {
                        xRow[meta.startIndex] = 0; // fallback to training mean (which is 0 after standardisation)
                    } else {
                        const num = Number(rawVal);
                        const scaled = (num - meta.mean) / meta.std;
                        xRow[meta.startIndex] = isNaN(scaled) ? 0 : Number(scaled.toFixed(5));
                    }
                }
            });

            return xRow;
        },

        /**
         * Pseudo-Random Number Generator (Mulberry32) for reproducible data splits & shuffling
         */
        createPRNG(seed) {
            if (seed === null || seed === undefined || seed === '' || isNaN(seed)) {
                return Math.random;
            }
            let s = Math.floor(Math.abs(Number(seed))) >>> 0;
            if (s === 0) s = 1;
            return function() {
                s |= 0; s = (s + 0x6D2B79F5) | 0;
                let t = Math.imul(s ^ (s >>> 15), 1 | s);
                t = t + Math.imul(t ^ (t >>> 7), 61 | t) ^ t;
                return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
            };
        },

        /**
         * Train / Test Split with deterministic random seed support
         */
        trainTestSplit(X, y, trainRatio = 0.8, isClassification = true, seed = null) {
            const total = X.length;
            const indices = Array.from({ length: total }, (_, i) => i);
            const rng = this.createPRNG(seed);

            // Deterministic seeded Fisher-Yates shuffle
            for (let i = indices.length - 1; i > 0; i--) {
                const j = Math.floor(rng() * (i + 1));
                [indices[i], indices[j]] = [indices[j], indices[i]];
            }

            let trainIndices = [];
            let testIndices = [];

            if (isClassification) {
                // Stratified split: partition by class
                const classIndices = {};
                indices.forEach(idx => {
                    const cls = y[idx];
                    if (!classIndices[cls]) classIndices[cls] = [];
                    classIndices[cls].push(idx);
                });

                Object.keys(classIndices).forEach(cls => {
                    const cIdxs = classIndices[cls];
                    const trainCount = Math.max(1, Math.round(cIdxs.length * trainRatio));
                    trainIndices.push(...cIdxs.slice(0, trainCount));
                    testIndices.push(...cIdxs.slice(trainCount));
                });
            } else {
                const trainCount = Math.round(total * trainRatio);
                trainIndices = indices.slice(0, trainCount);
                testIndices = indices.slice(trainCount);
            }

            // Ensure test has at least 1 sample if total >= 4
            if (testIndices.length === 0 && total >= 4) {
                testIndices.push(trainIndices.pop());
            }

            return {
                trainX: trainIndices.map(i => X[i]),
                trainY: trainIndices.map(i => y[i]),
                testX: testIndices.map(i => X[i]),
                testY: testIndices.map(i => y[i]),
                trainIndices,
                testIndices
            };
        },

        /**
         * Transforms a separate test dataset using training metadata,
         * extracting X_test and y_test strictly without data leakage.
         */
        prepareTestingDataset(testRows, prepMeta) {
            const { targetCol, isClassification, targetClassToIdx } = prepMeta;
            const X_test = [];
            const y_test = [];
            const validRows = [];
            let missingTargetCount = 0;
            let unknownClassCount = 0;

            testRows.forEach(row => {
                const rawTarget = row[targetCol];
                if (this.isMissing(rawTarget)) {
                    missingTargetCount++;
                    return;
                }

                let targetVal;
                if (isClassification) {
                    const strTarget = String(rawTarget).trim();
                    if (targetClassToIdx[strTarget] === undefined) {
                        unknownClassCount++;
                        return;
                    }
                    targetVal = targetClassToIdx[strTarget];
                } else {
                    targetVal = Number(rawTarget);
                    if (isNaN(targetVal)) {
                        missingTargetCount++;
                        return;
                    }
                }

                const xRow = this.transformSingleRow(row, prepMeta);
                X_test.push(xRow);
                y_test.push(targetVal);
                validRows.push(row);
            });

            return {
                X: X_test,
                y: y_test,
                validRows,
                totalEvaluated: X_test.length,
                totalProvided: testRows.length,
                missingTargetCount,
                unknownClassCount
            };
        },

        /**
         * =========================================================================
         * 7-STEP PIPELINE EXTENSIONS
         * Strictly ordered, leak-free statistical & machine learning pipeline
         * Conceived & designed by Dr G Narenthiran MB ChB BSc(MedSci)(Hons) FEBNS FRCS(SN)
         * =========================================================================
         */

        /**
         * Fits an imputer strictly on the training set only (zero data leakage)
         */
        fitStrictImputer(trainRows, featureCols, targetCol, columnConfigs, strategy = 'median') {
            const fillValues = {};
            const columnStats = {};

            // Strictly impute predictor feature columns only - clinical outcome/target is NEVER imputed
            const allCols = [...featureCols];

            allCols.forEach(col => {
                const conf = columnConfigs[col] || { type: 'continuous' };
                const validVals = [];
                const validNums = [];

                trainRows.forEach(r => {
                    const val = r[col];
                    if (!this.isMissing(val)) {
                        validVals.push(val);
                        const n = Number(val);
                        if (!isNaN(n)) validNums.push(n);
                    }
                });

                let fillVal = 0;
                if (conf.type === 'nominal') {
                    // Mode for nominal
                    const freq = {};
                    let maxCount = 0;
                    let topVal = 'Unknown';
                    validVals.forEach(v => {
                        const s = String(v).trim();
                        freq[s] = (freq[s] || 0) + 1;
                        if (freq[s] > maxCount) {
                            maxCount = freq[s];
                            topVal = s;
                        }
                    });
                    fillVal = topVal;
                    columnStats[col] = { mode: topVal, distinctCount: Object.keys(freq).length };
                } else {
                    // Continuous / Ordinal: Median or Mean
                    if (validNums.length > 0) {
                        const sum = validNums.reduce((a, b) => a + b, 0);
                        const mean = sum / validNums.length;
                        const sorted = [...validNums].sort((a, b) => a - b);
                        const mid = Math.floor(sorted.length / 2);
                        const median = sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
                        
                        fillVal = (strategy === 'mean') ? Number(mean.toFixed(4)) : Number(median.toFixed(4));
                        const variance = validNums.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / validNums.length;
                        columnStats[col] = {
                            mean: Number(mean.toFixed(4)),
                            median: Number(median.toFixed(4)),
                            std: Number(Math.sqrt(variance).toFixed(4)),
                            min: sorted[0],
                            max: sorted[sorted.length - 1],
                            validCount: validNums.length
                        };
                    } else {
                        fillVal = 0;
                        columnStats[col] = { mean: 0, median: 0, std: 1, min: 0, max: 0, validCount: 0 };
                    }
                }

                fillValues[col] = fillVal;
            });

            return {
                fillValues,
                columnStats,
                strategy,
                fittedSampleCount: trainRows.length
            };
        },

        /**
         * Transforms dataset rows using a pre-fitted strict imputer
         */
        transformWithStrictImputer(rows, imputer, columnConfigs) {
            let missingFilledCount = 0;
            const imputedRows = rows.map(r => {
                const newRow = { ...r };
                Object.keys(imputer.fillValues).forEach(col => {
                    if (this.isMissing(newRow[col])) {
                        newRow[col] = imputer.fillValues[col];
                        missingFilledCount++;
                    } else {
                        const conf = columnConfigs[col];
                        if (conf && (conf.type === 'continuous' || conf.type === 'ordinal')) {
                            const n = Number(newRow[col]);
                            if (!isNaN(n)) newRow[col] = n;
                        }
                    }
                });
                return newRow;
            });

            return {
                rows: imputedRows,
                missingFilledCount
            };
        },

        /**
         * Computes sample skewness to test for normality
         * g1 = [n / ((n-1)(n-2))] * sum(((x - mean)/s)^3)
         */
        calculateSkewness(arr) {
            const nums = arr.map(Number).filter(n => !isNaN(n));
            const n = nums.length;
            if (n < 3) return { skewness: 0, isNormal: true, mean: 0, std: 1 };

            const mean = nums.reduce((a, b) => a + b, 0) / n;
            const s2 = nums.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / (n - 1);
            const s = Math.sqrt(s2);
            if (s === 0) return { skewness: 0, isNormal: true, mean, std: 0 };

            const m3 = nums.reduce((a, b) => a + Math.pow(b - mean, 3), 0);
            const g1 = (n * m3) / ((n - 1) * (n - 2) * Math.pow(s, 3));
            const isNormal = Math.abs(g1) <= 1.0;

            return {
                skewness: Number(g1.toFixed(3)),
                isNormal,
                mean: Number(mean.toFixed(3)),
                std: Number(s.toFixed(3))
            };
        },

        testNormality(arr) {
            return this.calculateSkewness(arr);
        },

        /**
         * Pearson Product-Moment Correlation Coefficient
         */
        calculatePearson(x, y) {
            const n = Math.min(x.length, y.length);
            if (n < 2) return 0;
            const xMean = x.reduce((a, b) => a + b, 0) / n;
            const yMean = y.reduce((a, b) => a + b, 0) / n;

            let num = 0;
            let denX = 0;
            let denY = 0;

            for (let i = 0; i < n; i++) {
                const dx = x[i] - xMean;
                const dy = y[i] - yMean;
                num += dx * dy;
                denX += dx * dx;
                denY += dy * dy;
            }

            const den = Math.sqrt(denX * denY);
            if (den === 0) return 0;
            return Number((num / den).toFixed(4));
        },

        /**
         * Spearman Rank-Order Correlation Coefficient
         */
        calculateSpearman(x, y) {
            const n = Math.min(x.length, y.length);
            if (n < 2) return 0;

            function getRanks(arr) {
                const indexed = arr.map((val, idx) => ({ val, idx }));
                indexed.sort((a, b) => a.val - b.val);
                const ranks = new Array(n);
                let i = 0;
                while (i < n) {
                    let j = i;
                    while (j < n - 1 && indexed[j + 1].val === indexed[j].val) {
                        j++;
                    }
                    const avgRank = (i + j + 2) / 2; // 1-based average rank
                    for (let k = i; k <= j; k++) {
                        ranks[indexed[k].idx] = avgRank;
                    }
                    i = j + 1;
                }
                return ranks;
            }

            const xRanks = getRanks(x);
            const yRanks = getRanks(y);
            return this.calculatePearson(xRanks, yRanks);
        },

        /**
         * Checks pairwise continuous variable correlation using Training set
         * Applies Pearson when both are normal; Spearman when one/both are skewed
         */
        checkPairwiseCorrelations(trainRows, continuousCols, threshold = 0.8) {
            const normalityMap = {};
            continuousCols.forEach(col => {
                const vals = trainRows.map(r => Number(r[col])).filter(n => !isNaN(n));
                normalityMap[col] = this.calculateSkewness(vals);
            });

            const pairs = [];
            for (let i = 0; i < continuousCols.length; i++) {
                for (let j = i + 1; j < continuousCols.length; j++) {
                    const cA = continuousCols[i];
                    const cB = continuousCols[j];
                    const valsA = trainRows.map(r => Number(r[cA]));
                    const valsB = trainRows.map(r => Number(r[cB]));

                    const normA = normalityMap[cA];
                    const normB = normalityMap[cB];
                    const bothNormal = normA.isNormal && normB.isNormal;
                    const method = bothNormal ? 'Pearson (r)' : 'Spearman (ρ)';
                    const coeff = bothNormal ? this.calculatePearson(valsA, valsB) : this.calculateSpearman(valsA, valsB);
                    const absCoeff = Math.abs(coeff);

                    pairs.push({
                        var1: cA,
                        var2: cB,
                        method,
                        coeff,
                        absCoeff,
                        norm1: normA,
                        norm2: normB,
                        bothNormal,
                        isHigh: absCoeff >= threshold
                    });
                }
            }

            // Sort descending by absolute correlation
            pairs.sort((a, b) => b.absCoeff - a.absCoeff);

            return {
                normalityMap,
                allPairs: pairs,
                highPairs: pairs.filter(p => p.isHigh)
            };
        },

        /**
         * Matrix Inversion using Gauss-Jordan elimination with partial pivoting
         */
        invertMatrix(A) {
            const n = A.length;
            const M = A.map((row, i) => {
                const r = [...row];
                const id = new Array(n).fill(0);
                id[i] = 1;
                return r.concat(id);
            });

            for (let i = 0; i < n; i++) {
                let maxRow = i;
                for (let k = i + 1; k < n; k++) {
                    if (Math.abs(M[k][i]) > Math.abs(M[maxRow][i])) maxRow = k;
                }
                [M[i], M[maxRow]] = [M[maxRow], M[i]];

                const pivot = M[i][i];
                if (Math.abs(pivot) < 1e-12) return null; // Singular matrix
                for (let j = 0; j < 2 * n; j++) M[i][j] /= pivot;

                for (let k = 0; k < n; k++) {
                    if (k === i) continue;
                    const factor = M[k][i];
                    for (let j = 0; j < 2 * n; j++) {
                        M[k][j] -= factor * M[i][j];
                    }
                }
            }
            return M.map(row => row.slice(n));
        },

        /**
         * Computes Variance Inflation Factor (VIF) on Training set features
         * One-hot encodes categoricals (dropping reference column) and computes R^2
         */
        calculateVIF(trainRows, featureCols, columnConfigs) {
            // Build design matrix with reference category dropped for nominal features
            const encodedCols = [];
            const colMapping = {}; // maps encoded column name back to original feature name

            featureCols.forEach(col => {
                const conf = columnConfigs[col];
                if (conf && conf.type === 'nominal') {
                    const distinct = Array.from(new Set(trainRows.map(r => String(r[col])))).sort();
                    // Drop reference column (first category) to avoid dummy trap
                    const nonRef = distinct.slice(1);
                    nonRef.forEach(cat => {
                        const dummyName = `${col}_is_${cat}`;
                        encodedCols.push({
                            name: dummyName,
                            originalCol: col,
                            type: 'dummy',
                            extract: (r) => String(r[col]) === cat ? 1 : 0
                        });
                    });
                } else {
                    // Continuous or Ordinal
                    encodedCols.push({
                        name: col,
                        originalCol: col,
                        type: 'numeric',
                        extract: (r) => {
                            const n = Number(r[col]);
                            return isNaN(n) ? 0 : n;
                        }
                    });
                }
            });

            if (encodedCols.length < 2) {
                return featureCols.map(col => ({
                    feature: col,
                    vif: 1.0,
                    rSquared: 0,
                    highVIF: false,
                    dummyCount: 1
                }));
            }

            const N = trainRows.length;
            const P = encodedCols.length;

            // Extract numeric values and standardize columns
            const X_raw = Array.from({ length: N }, () => new Array(P).fill(0));
            for (let j = 0; j < P; j++) {
                const extractor = encodedCols[j].extract;
                let sum = 0;
                for (let i = 0; i < N; i++) {
                    const val = extractor(trainRows[i]);
                    X_raw[i][j] = val;
                    sum += val;
                }
                const mean = sum / N;
                let varSum = 0;
                for (let i = 0; i < N; i++) varSum += Math.pow(X_raw[i][j] - mean, 2);
                const std = Math.sqrt(varSum / N) || 1;
                for (let i = 0; i < N; i++) {
                    X_raw[i][j] = (X_raw[i][j] - mean) / std;
                }
            }

            // Calculate VIF for each encoded column via regression on all other columns
            const encodedVIFs = [];
            for (let j = 0; j < P; j++) {
                // Dependent var y = X[:, j]
                const y = X_raw.map(row => row[j]);
                // Predictors Z = X[:, != j] with intercept
                const Z = X_raw.map(row => [1, ...row.filter((_, idx) => idx !== j)]);
                const dimZ = Z[0].length;

                // Compute Z^T * Z + lambda * I
                const ZtZ = Array.from({ length: dimZ }, () => new Array(dimZ).fill(0));
                const Zty = new Array(dimZ).fill(0);

                for (let i = 0; i < N; i++) {
                    const zRow = Z[i];
                    const yVal = y[i];
                    for (let a = 0; a < dimZ; a++) {
                        Zty[a] += zRow[a] * yVal;
                        for (let b = 0; b < dimZ; b++) {
                            ZtZ[a][b] += zRow[a] * zRow[b];
                        }
                    }
                }

                // Add small ridge lambda to guarantee well-conditioned inversion
                for (let a = 0; a < dimZ; a++) ZtZ[a][a] += 1e-4;

                const invZtZ = this.invertMatrix(ZtZ);
                let r2 = 0;
                if (invZtZ) {
                    const beta = new Array(dimZ).fill(0);
                    for (let a = 0; a < dimZ; a++) {
                        for (let b = 0; b < dimZ; b++) {
                            beta[a] += invZtZ[a][b] * Zty[b];
                        }
                    }

                    let ssTot = 0;
                    let ssRes = 0;
                    const yMean = y.reduce((a, b) => a + b, 0) / N;

                    for (let i = 0; i < N; i++) {
                        let yHat = 0;
                        for (let a = 0; a < dimZ; a++) yHat += Z[i][a] * beta[a];
                        ssTot += Math.pow(y[i] - yMean, 2);
                        ssRes += Math.pow(y[i] - yHat, 2);
                    }

                    r2 = ssTot > 0 ? Math.max(0, 1 - (ssRes / ssTot)) : 0;
                }

                let vif = r2 >= 0.999 ? 999.0 : 1 / Math.max(0.001, 1 - r2);
                encodedVIFs.push({
                    encodedName: encodedCols[j].name,
                    originalCol: encodedCols[j].originalCol,
                    rSquared: Number(r2.toFixed(4)),
                    vif: Number(vif.toFixed(2))
                });
            }

            // Aggregate back to original clinical feature level (taking the max VIF across its dummy levels)
            const featureVIFMap = {};
            encodedVIFs.forEach(item => {
                const col = item.originalCol;
                if (!featureVIFMap[col] || item.vif > featureVIFMap[col].vif) {
                    featureVIFMap[col] = {
                        feature: col,
                        vif: item.vif,
                        rSquared: item.rSquared,
                        type: columnConfigs[col]?.type || 'continuous',
                        dummyCount: (featureVIFMap[col]?.dummyCount || 0) + 1,
                        highVIF: item.vif > 5.0
                    };
                } else {
                    featureVIFMap[col].dummyCount++;
                }
            });

            const results = Object.values(featureVIFMap);
            results.sort((a, b) => b.vif - a.vif);
            return results;
        },

        /**
         * Accurate Lanczos approximation for log(Gamma(x))
         */
        logGamma(x) {
            const cof = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
            let y = x;
            let tmp = x + 5.5;
            tmp -= (x + 0.5) * Math.log(tmp);
            let ser = 1.000000000190015;
            for (let j = 0; j <= 5; j++) ser += cof[j] / ++y;
            return -tmp + Math.log(2.5066282746310005 * ser / x);
        },

        /**
         * Incomplete Gamma function P(a, x) for Chi-Square p-values
         */
        gammp(a, x) {
            if (x <= 0 || a <= 0) return 0;
            const ITMAX = 100, EPS = 3e-7, FPMIN = 1e-30;
            if (x < a + 1) {
                // Series representation
                let sum = 1 / a, del = sum, ap = a;
                for (let n = 1; n <= ITMAX; n++) {
                    ap += 1;
                    del *= x / ap;
                    sum += del;
                    if (Math.abs(del) < Math.abs(sum) * EPS) {
                        return sum * Math.exp(-x + a * Math.log(x) - this.logGamma(a));
                    }
                }
                return sum * Math.exp(-x + a * Math.log(x) - this.logGamma(a));
            } else {
                // Continued fraction representation
                let b = x + 1 - a, c = 1 / FPMIN, d = 1 / b, h = d;
                for (let i = 1; i <= ITMAX; i++) {
                    const an = -i * (i - a);
                    b += 2;
                    d = an * d + b;
                    if (Math.abs(d) < FPMIN) d = FPMIN;
                    c = b + an / c;
                    if (Math.abs(c) < FPMIN) c = FPMIN;
                    d = 1 / d;
                    const del = d * c;
                    h *= del;
                    if (Math.abs(del - 1) < EPS) break;
                }
                return 1 - (Math.exp(-x + a * Math.log(x) - this.logGamma(a)) * h);
            }
        },

        chiSquarePValue(chiSq, df) {
            if (chiSq <= 0 || df <= 0) return 1.0;
            const p = 1 - this.gammp(df / 2, chiSq / 2);
            return Math.max(0, Math.min(1, p));
        },

        /**
         * Incomplete Beta function I_x(a, b) for ANOVA / F-test p-values
         */
        betai(a, b, x) {
            if (x <= 0) return 0;
            if (x >= 1) return 1;
            const ITMAX = 100, EPS = 3e-7, FPMIN = 1e-30;
            const bt = Math.exp(this.logGamma(a + b) - this.logGamma(a) - this.logGamma(b) + a * Math.log(x) + b * Math.log(1 - x));

            function betacf(aa, bb, xx) {
                const qab = aa + bb, qap = aa + 1, qam = aa - 1;
                let c = 1, d = 1 - qab * xx / qap;
                if (Math.abs(d) < FPMIN) d = FPMIN;
                d = 1 / d;
                let h = d;
                for (let m = 1; m <= ITMAX; m++) {
                    const m2 = 2 * m;
                    let num = m * (bb - m) * xx / ((qam + m2) * (aa + m2));
                    d = 1 + num * d;
                    if (Math.abs(d) < FPMIN) d = FPMIN;
                    c = 1 + num / c;
                    if (Math.abs(c) < FPMIN) c = FPMIN;
                    d = 1 / d;
                    h *= d * c;
                    num = -(aa + m) * (qab + m) * xx / ((aa + m2) * (qap + m2));
                    d = 1 + num * d;
                    if (Math.abs(d) < FPMIN) d = FPMIN;
                    c = 1 + num / c;
                    if (Math.abs(c) < FPMIN) c = FPMIN;
                    d = 1 / d;
                    const del = d * c;
                    h *= del;
                    if (Math.abs(del - 1) < EPS) break;
                }
                return h;
            }

            if (x < (a + 1) / (a + b + 2)) {
                return (bt * betacf(a, b, x)) / a;
            } else {
                return 1 - ((bt * betacf(b, a, 1 - x)) / b);
            }
        },

        fTestPValue(fStat, df1, df2) {
            if (fStat <= 0 || df1 <= 0 || df2 <= 0) return 1.0;
            const x = df2 / (df2 + df1 * fStat);
            return Math.max(0, Math.min(1, this.betai(df2 / 2, df1 / 2, x)));
        },

        /**
         * Univariate statistical baseline tests (Step 4)
         * - ANOVA F-test for continuous features vs categorical target
         * - Chi-Square test for categorical features vs categorical target
         * Returns summary table sorted by p-value (lowest to highest)
         */
        runUnivariateTesting(rows, survivingCols, targetCol, columnConfigs) {
            const isTargetContinuous = columnConfigs[targetCol]?.type === 'continuous';
            const results = [];

            survivingCols.forEach(col => {
                const isFeatureContinuous = columnConfigs[col]?.type === 'continuous' || columnConfigs[col]?.type === 'ordinal';

                if (!isTargetContinuous) {
                    // Categorical Target (e.g. Binary Classification)
                    const targetClasses = Array.from(new Set(rows.map(r => String(r[targetCol])))).sort();
                    const K = targetClasses.length;

                    if (isFeatureContinuous) {
                        // One-Way ANOVA F-test
                        const groups = {};
                        targetClasses.forEach(c => groups[c] = []);

                        // For text ordinal features, build rank mapping first
                        let rankMap = null;
                        if (columnConfigs[col]?.type === 'ordinal') {
                            const distinctVals = Array.from(new Set(rows.map(r => r[col]))).filter(v => !this.isMissing(v));
                            if (distinctVals.some(v => isNaN(Number(v)))) {
                                rankMap = {};
                                const sorted = [...distinctVals];
                                const clinicalScales = ['none', 'mild', 'moderate', 'severe', 'absent', 'present', 'low', 'medium', 'high', 'very high', 'no', 'yes'];
                                const lower = sorted.map(s => String(s).toLowerCase().trim());
                                if (lower.every(r => clinicalScales.includes(r))) {
                                    sorted.sort((a, b) => clinicalScales.indexOf(String(a).toLowerCase().trim()) - clinicalScales.indexOf(String(b).toLowerCase().trim()));
                                } else {
                                    sorted.sort();
                                }
                                sorted.forEach((v, idx) => { rankMap[String(v)] = idx; });
                            }
                        }

                        rows.forEach(r => {
                            let val;
                            if (rankMap) {
                                val = rankMap[String(r[col])];
                            } else {
                                val = Number(r[col]);
                            }
                            if (!isNaN(val) && val !== null && val !== undefined) {
                                groups[String(r[targetCol])].push(val);
                            }
                        });

                        const N = rows.length;
                        let grandSum = 0;
                        let totalValid = 0;
                        targetClasses.forEach(c => {
                            groups[c].forEach(v => { grandSum += v; totalValid++; });
                        });
                        const grandMean = grandSum / (totalValid || 1);

                        let ssBetween = 0;
                        let ssWithin = 0;
                        targetClasses.forEach(c => {
                            const grp = groups[c];
                            const n_c = grp.length;
                            if (n_c > 0) {
                                const grpMean = grp.reduce((a, b) => a + b, 0) / n_c;
                                ssBetween += n_c * Math.pow(grpMean - grandMean, 2);
                                grp.forEach(v => { ssWithin += Math.pow(v - grpMean, 2); });
                            }
                        });

                        const df1 = K - 1;
                        const df2 = Math.max(1, totalValid - K);
                        const msBetween = ssBetween / (df1 || 1);
                        const msWithin = ssWithin / (df2 || 1);
                        const fStat = msWithin > 0 ? (msBetween / msWithin) : 0;
                        const pVal = this.fTestPValue(fStat, df1, df2);

                        results.push({
                            feature: col,
                            scale: 'Continuous / Ordinal',
                            testName: K === 2 ? "Two-Sample t-test / ANOVA (F)" : "One-Way ANOVA (F-test)",
                            statisticLabel: 'F-stat',
                            statisticValue: Number(fStat.toFixed(3)),
                            df: `${df1}, ${df2}`,
                            pValue: Number(pVal.toFixed(5)),
                            significant: pVal < 0.05
                        });
                    } else {
                        // Categorical vs Categorical: Chi-Square Test
                        const categories = Array.from(new Set(rows.map(r => String(r[col])))).sort();
                        const R = categories.length;

                        // Contingency Table
                        const obs = Array.from({ length: R }, () => new Array(K).fill(0));
                        const rowTotals = new Array(R).fill(0);
                        const colTotals = new Array(K).fill(0);
                        let totalN = 0;

                        rows.forEach(r => {
                            const rIdx = categories.indexOf(String(r[col]));
                            const cIdx = targetClasses.indexOf(String(r[targetCol]));
                            if (rIdx !== -1 && cIdx !== -1) {
                                obs[rIdx][cIdx]++;
                                rowTotals[rIdx]++;
                                colTotals[cIdx]++;
                                totalN++;
                            }
                        });

                        let chiSq = 0;
                        for (let i = 0; i < R; i++) {
                            for (let j = 0; j < K; j++) {
                                const expected = (rowTotals[i] * colTotals[j]) / (totalN || 1);
                                if (expected > 0) {
                                    chiSq += Math.pow(obs[i][j] - expected, 2) / expected;
                                }
                            }
                        }

                        const df = Math.max(1, (R - 1) * (K - 1));
                        const pVal = this.chiSquarePValue(chiSq, df);

                        results.push({
                            feature: col,
                            scale: 'Nominal Categorical',
                            testName: "Pearson Chi-Square (χ²)",
                            statisticLabel: 'χ² stat',
                            statisticValue: Number(chiSq.toFixed(3)),
                            df: `${df}`,
                            pValue: Number(pVal.toFixed(5)),
                            significant: pVal < 0.05
                        });
                    }
                } else {
                    // Continuous Target (Regression)
                    if (isFeatureContinuous) {
                        const valsX = rows.map(r => Number(r[col]));
                        const valsY = rows.map(r => Number(r[targetCol]));
                        const rCoeff = this.calculatePearson(valsX, valsY);
                        const n = rows.length;
                        const tStat = Math.abs(rCoeff) < 1 ? rCoeff * Math.sqrt((n - 2) / (1 - rCoeff * rCoeff)) : 999;
                        const fEquiv = tStat * tStat;
                        const pVal = this.fTestPValue(fEquiv, 1, n - 2);

                        results.push({
                            feature: col,
                            scale: 'Continuous',
                            testName: "Pearson Linear Correlation (t-test)",
                            statisticLabel: 'r / t-stat',
                            statisticValue: Number(rCoeff.toFixed(3)),
                            df: `${n - 2}`,
                            pValue: Number(pVal.toFixed(5)),
                            significant: pVal < 0.05
                        });
                    } else {
                        // Categorical feature vs Continuous Target: ANOVA
                        const categories = Array.from(new Set(rows.map(r => String(r[col])))).sort();
                        const K = categories.length;
                        const groups = {};
                        categories.forEach(c => groups[c] = []);
                        rows.forEach(r => {
                            const y = Number(r[targetCol]);
                            if (!isNaN(y)) groups[String(r[col])].push(y);
                        });

                        const N = rows.length;
                        const grandMean = rows.reduce((a, b) => a + Number(b[targetCol]), 0) / N;
                        let ssBetween = 0;
                        let ssWithin = 0;

                        categories.forEach(c => {
                            const grp = groups[c];
                            const n_c = grp.length;
                            if (n_c > 0) {
                                const grpMean = grp.reduce((a, b) => a + b, 0) / n_c;
                                ssBetween += n_c * Math.pow(grpMean - grandMean, 2);
                                grp.forEach(v => { ssWithin += Math.pow(v - grpMean, 2); });
                            }
                        });

                        const df1 = K - 1;
                        const df2 = Math.max(1, N - K);
                        const msBetween = ssBetween / (df1 || 1);
                        const msWithin = ssWithin / (df2 || 1);
                        const fStat = msWithin > 0 ? (msBetween / msWithin) : 0;
                        const pVal = this.fTestPValue(fStat, df1, df2);

                        results.push({
                            feature: col,
                            scale: 'Nominal Categorical',
                            testName: "One-Way ANOVA (F-test)",
                            statisticLabel: 'F-stat',
                            statisticValue: Number(fStat.toFixed(3)),
                            df: `${df1}, ${df2}`,
                            pValue: Number(pVal.toFixed(5)),
                            significant: pVal < 0.05
                        });
                    }
                }
            });

            // Sort ascending by p-value (lowest to highest) as specified in Step 4
            results.sort((a, b) => a.pValue - b.pValue);
            return results;
        },

        /**
         * Step 5: Imbalance Handling using SMOTENC
         * Applied strictly to the Training set ONLY if minority class < 30%
         */
        applySMOTENC(trainRows, featureCols, targetCol, columnConfigs, seed = 42) {
            const isClassification = columnConfigs[targetCol]?.type !== 'continuous';
            if (!isClassification) {
                return {
                    balancedRows: trainRows,
                    applied: false,
                    reason: 'Target is continuous (Regression); SMOTENC applies to classification.'
                };
            }

            const classCounts = {};
            trainRows.forEach(r => {
                const c = String(r[targetCol]);
                classCounts[c] = (classCounts[c] || 0) + 1;
            });

            const classes = Object.keys(classCounts);
            if (classes.length < 2) {
                return {
                    balancedRows: trainRows,
                    applied: false,
                    reason: 'Single class present in training data.'
                };
            }

            const total = trainRows.length;
            let minorityClass = classes[0];
            let majorityClass = classes[0];
            let minCount = Infinity;
            let maxCount = -Infinity;

            classes.forEach(c => {
                if (classCounts[c] < minCount) {
                    minCount = classCounts[c];
                    minorityClass = c;
                }
                if (classCounts[c] > maxCount) {
                    maxCount = classCounts[c];
                    majorityClass = c;
                }
            });

            const imbalanceRatio = minCount / (maxCount || 1);
            const threshold = 0.60; // Imbalance flagged if minority is < 60% of majority class

            if (imbalanceRatio >= threshold) {
                return {
                    balancedRows: trainRows,
                    applied: false,
                    minorityClass,
                    imbalanceRatio: Number((imbalanceRatio * 100).toFixed(1)),
                    classCounts,
                    reason: `Minority class is ${(imbalanceRatio * 100).toFixed(1)}% of majority class (>= 60% threshold). Cohort is sufficiently balanced.`
                };
            }

            // Identify continuous and categorical columns
            const contCols = featureCols.filter(c => columnConfigs[c]?.type === 'continuous' || columnConfigs[c]?.type === 'ordinal');
            const catCols = featureCols.filter(c => columnConfigs[c]?.type === 'nominal');

            // Extract minority samples
            const minorityRows = trainRows.filter(r => String(r[targetCol]) === minorityClass);
            const m = minorityRows.length;

            if (m < 2) {
                return {
                    balancedRows: trainRows,
                    applied: false,
                    reason: 'Insufficient minority samples for SMOTENC neighbor search.'
                };
            }

            // Compute standard deviations for continuous features among minority class
            const stdMap = {};
            contCols.forEach(col => {
                const vals = minorityRows.map(r => Number(r[col])).filter(n => !isNaN(n));
                const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
                const v = vals.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / (vals.length || 1);
                const s = Math.sqrt(v);
                stdMap[col] = s > 0 ? s : 1.0;
            });

            // Since continuous distances are standardized by stdMap, each has unit variance.
            // A categorical mismatch penalty is 1.0 (equivalent to 1 standard deviation mismatch).
            const catPenalty = 1.0;

            // Compute number of synthetic samples needed to bring minority to majority (balanced)
            const numSynthNeeded = maxCount - minCount;
            const rng = this.createPRNG(seed);
            const k = Math.min(5, m - 1);
            const syntheticRows = [];

            for (let sIdx = 0; sIdx < numSynthNeeded; sIdx++) {
                // Randomly pick a minority sample
                const i = Math.floor(rng() * m);
                const baseRow = minorityRows[i];

                // Find k-nearest minority neighbors
                const dists = [];
                for (let j = 0; j < m; j++) {
                    if (j === i) continue;
                    const otherRow = minorityRows[j];
                    let distSq = 0;

                    contCols.forEach(col => {
                        const diff = (Number(baseRow[col]) - Number(otherRow[col])) / (stdMap[col] || 1);
                        distSq += diff * diff;
                    });

                    catCols.forEach(col => {
                        if (String(baseRow[col]) !== String(otherRow[col])) {
                            distSq += catPenalty;
                        }
                    });

                    dists.push({ row: otherRow, dist: Math.sqrt(distSq) });
                }

                dists.sort((a, b) => a.dist - b.dist);
                const neighbors = dists.slice(0, k);

                // Pick one neighbor at random
                const chosenNeighbor = neighbors[Math.floor(rng() * neighbors.length)].row;

                // Synthesize new row
                const newRow = {};
                // Continuous features: interpolation; Ordinal features: integer rounding
                const lambda = rng();
                contCols.forEach(col => {
                    const v1 = Number(baseRow[col]);
                    const v2 = Number(chosenNeighbor[col]);
                    if (columnConfigs[col]?.type === 'ordinal') {
                        newRow[col] = Math.round(v1 + lambda * (v2 - v1));
                    } else {
                        newRow[col] = Number((v1 + lambda * (v2 - v1)).toFixed(4));
                    }
                });

                // Categorical features: mode among the k neighbors
                catCols.forEach(col => {
                    const votes = {};
                    neighbors.forEach(nbr => {
                        const v = String(nbr.row[col]);
                        votes[v] = (votes[v] || 0) + 1;
                    });
                    let bestCat = String(baseRow[col]);
                    let topVotes = 0;
                    Object.keys(votes).forEach(v => {
                        if (votes[v] > topVotes) {
                            topVotes = votes[v];
                            bestCat = v;
                        }
                    });
                    newRow[col] = bestCat;
                });

                newRow[targetCol] = minorityClass;
                syntheticRows.push(newRow);
            }

            const balancedRows = [...trainRows, ...syntheticRows];
            const newCounts = { ...classCounts, [minorityClass]: minCount + numSynthNeeded };

            return {
                balancedRows,
                applied: true,
                minorityClass,
                minorityPctBefore: Number((minorityPct * 100).toFixed(1)),
                minorityPctAfter: Number(((minCount + numSynthNeeded) / balancedRows.length * 100).toFixed(1)),
                originalCounts: classCounts,
                balancedCounts: newCounts,
                syntheticCountGenerated: numSynthNeeded,
                reason: `Minority class was ${(minorityPct * 100).toFixed(1)}% (< 30%). SMOTENC synthesized ${numSynthNeeded} balanced training instances.`
            };
        }
    };

    window.DataEngine = DataEngine;
})(window);
