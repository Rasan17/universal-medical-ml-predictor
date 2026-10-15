/**
 * NPH ML/Neuralnet Predictor - Machine Learning & Deep Learning Engine
 * Provides Deep Neural Networks (TensorFlow.js + Pure JS Fallback), Random Forests,
 * Logistic/Softmax Regression, Ridge Regression, Support Vector Machines, k-NN,
 * and Gaussian Naive Bayes with comprehensive evaluation metrics.
 * 
 * Conceived, designed and developed by Dr G Narenthiran MB ChB BSc(MedSci)(Hons) FEBNS FRCS(SN)
 */

(function (window) {
    'use strict';

    /**
     * Pseudo-Random Number Generator (Mulberry32) for deterministic reproducible modeling
     */
    function createPRNG(seed) {
        if (seed === null || seed === undefined || seed === '' || isNaN(seed)) {
            return Math.random;
        }
        let s = Math.floor(Math.abs(Number(seed))) >>> 0;
        if (s === 0) s = 1;
        return function () {
            s |= 0; s = (s + 0x6D2B79F5) | 0;
            let t = Math.imul(s ^ (s >>> 15), 1 | s);
            t = t + Math.imul(t ^ (t >>> 7), 61 | t) ^ t;
            return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
        };
    }

    // ==========================================
    // 1. PURE JS DECISION TREE & RANDOM FOREST
    // ==========================================
    class DecisionTreeNode {
        constructor(isLeaf = false, value = null, featureIdx = null, threshold = null, left = null, right = null) {
            this.isLeaf = isLeaf;
            this.value = value; // For classification: class or prob dict; for regression: mean value
            this.featureIdx = featureIdx;
            this.threshold = threshold;
            this.left = left;
            this.right = right;
        }
    }

    class DecisionTree {
        constructor(isClassification = true, maxDepth = 6, minSamplesSplit = 2, maxFeatures = null, rng = Math.random) {
            this.isClassification = isClassification;
            this.maxDepth = maxDepth;
            this.minSamplesSplit = minSamplesSplit;
            this.maxFeatures = maxFeatures;
            this.rng = rng || Math.random;
            this.root = null;
            this.featureImportances = {};
        }

        fit(X, y) {
            const numFeatures = X[0].length;
            this.featureImportances = new Array(numFeatures).fill(0);
            this.root = this._buildTree(X, y, 0);
            // Normalize feature importances
            const sum = this.featureImportances.reduce((a, b) => a + b, 0);
            if (sum > 0) {
                this.featureImportances = this.featureImportances.map(v => v / sum);
            }
        }

        _buildTree(X, y, depth) {
            const numSamples = X.length;
            const numFeatures = X[0].length;

            if (this.isClassification) {
                const uniqueClasses = [...new Set(y)];
                if (uniqueClasses.length === 1 || depth >= this.maxDepth || numSamples < this.minSamplesSplit) {
                    const classCounts = {};
                    y.forEach(c => classCounts[c] = (classCounts[c] || 0) + 1);
                    let majority = y[0];
                    let maxCount = 0;
                    Object.keys(classCounts).forEach(c => {
                        if (classCounts[c] > maxCount) {
                            maxCount = classCounts[c];
                            majority = Number(c);
                        }
                    });
                    return new DecisionTreeNode(true, majority);
                }
            } else {
                const mean = y.reduce((a, b) => a + b, 0) / (numSamples || 1);
                if (depth >= this.maxDepth || numSamples < this.minSamplesSplit) {
                    return new DecisionTreeNode(true, mean);
                }
            }

            // Choose random subset of features
            let featureIndices = Array.from({ length: numFeatures }, (_, i) => i);
            if (this.maxFeatures && this.maxFeatures < numFeatures) {
                featureIndices.sort(() => this.rng() - 0.5);
                featureIndices = featureIndices.slice(0, this.maxFeatures);
            }

            let bestGain = -Infinity;
            let bestFeature = null;
            let bestThreshold = null;
            let bestSplits = null;

            const currentScore = this.isClassification ? this._gini(y) : this._variance(y);

            featureIndices.forEach(featIdx => {
                const values = X.map(r => r[featIdx]);
                // Sample potential split thresholds
                const uniqueVals = [...new Set(values)].sort((a, b) => a - b);
                if (uniqueVals.length <= 1) return;

                const step = Math.max(1, Math.floor(uniqueVals.length / 10));
                for (let i = 0; i < uniqueVals.length - 1; i += step) {
                    const threshold = (uniqueVals[i] + uniqueVals[i + 1]) / 2;
                    const leftX = [], leftY = [], rightX = [], rightY = [];

                    for (let s = 0; s < numSamples; s++) {
                        if (X[s][featIdx] <= threshold) {
                            leftX.push(X[s]);
                            leftY.push(y[s]);
                        } else {
                            rightX.push(X[s]);
                            rightY.push(y[s]);
                        }
                    }

                    if (leftY.length === 0 || rightY.length === 0) continue;

                    let gain = 0;
                    if (this.isClassification) {
                        const leftGini = this._gini(leftY);
                        const rightGini = this._gini(rightY);
                        gain = currentScore - ((leftY.length / numSamples) * leftGini + (rightY.length / numSamples) * rightGini);
                    } else {
                        const leftVar = this._variance(leftY);
                        const rightVar = this._variance(rightY);
                        gain = currentScore - ((leftY.length / numSamples) * leftVar + (rightY.length / numSamples) * rightVar);
                    }

                    if (gain > bestGain) {
                        bestGain = gain;
                        bestFeature = featIdx;
                        bestThreshold = threshold;
                        bestSplits = { leftX, leftY, rightX, rightY };
                    }
                }
            });

            if (bestGain <= 0 || !bestSplits) {
                if (this.isClassification) {
                    const counts = {};
                    y.forEach(c => counts[c] = (counts[c] || 0) + 1);
                    let majority = y[0];
                    let maxCount = 0;
                    Object.keys(counts).forEach(c => {
                        if (counts[c] > maxCount) {
                            maxCount = counts[c];
                            majority = Number(c);
                        }
                    });
                    return new DecisionTreeNode(true, majority);
                } else {
                    return new DecisionTreeNode(true, y.reduce((a, b) => a + b, 0) / (numSamples || 1));
                }
            }

            this.featureImportances[bestFeature] += bestGain * numSamples;

            const leftChild = this._buildTree(bestSplits.leftX, bestSplits.leftY, depth + 1);
            const rightChild = this._buildTree(bestSplits.rightX, bestSplits.rightY, depth + 1);

            return new DecisionTreeNode(false, null, bestFeature, bestThreshold, leftChild, rightChild);
        }

        _gini(y) {
            const counts = {};
            y.forEach(c => counts[c] = (counts[c] || 0) + 1);
            let sumSq = 0;
            const n = y.length;
            Object.values(counts).forEach(cnt => {
                const p = cnt / n;
                sumSq += p * p;
            });
            return 1 - sumSq;
        }

        _variance(y) {
            const n = y.length;
            if (n === 0) return 0;
            const mean = y.reduce((a, b) => a + b, 0) / n;
            return y.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / n;
        }

        predictSample(x) {
            let node = this.root;
            while (!node.isLeaf) {
                if (x[node.featureIdx] <= node.threshold) {
                    node = node.left;
                } else {
                    node = node.right;
                }
            }
            return node.value;
        }
    }

    class RandomForest {
        constructor(isClassification = true, numTrees = 25, maxDepth = 5, minSamplesSplit = 2, featureRatio = 'sqrt', seed = null) {
            this.isClassification = isClassification;
            this.numTrees = numTrees;
            this.maxDepth = maxDepth;
            this.minSamplesSplit = minSamplesSplit;
            this.featureRatio = featureRatio;
            this.seed = seed !== null && seed !== undefined && seed !== '' ? Number(seed) : null;
            this.rng = createPRNG(this.seed);
            this.trees = [];
            this.featureImportances = [];
        }

        fit(X, y) {
            const numSamples = X.length;
            const numFeatures = X[0].length;

            let maxFeatures = Math.max(1, Math.floor(this.isClassification ? Math.sqrt(numFeatures) : numFeatures / 3));
            if (this.featureRatio === 'all') {
                maxFeatures = numFeatures;
            } else if (this.featureRatio === 'log2') {
                maxFeatures = Math.max(1, Math.floor(Math.log2(numFeatures)));
            } else if (this.featureRatio === '0.5') {
                maxFeatures = Math.max(1, Math.floor(numFeatures * 0.5));
            } else if (this.featureRatio === '0.75') {
                maxFeatures = Math.max(1, Math.floor(numFeatures * 0.75));
            }

            this.trees = [];
            const aggregatedImportances = new Array(numFeatures).fill(0);

            for (let t = 0; t < this.numTrees; t++) {
                // Bootstrap sample with replacement using seeded PRNG
                const sampleX = [];
                const sampleY = [];
                for (let i = 0; i < numSamples; i++) {
                    const randIdx = Math.floor(this.rng() * numSamples);
                    sampleX.push(X[randIdx]);
                    sampleY.push(y[randIdx]);
                }

                const tree = new DecisionTree(this.isClassification, this.maxDepth, this.minSamplesSplit, maxFeatures, this.rng);
                tree.fit(sampleX, sampleY);
                this.trees.push(tree);

                tree.featureImportances.forEach((imp, fIdx) => {
                    aggregatedImportances[fIdx] += imp;
                });
            }

            const total = aggregatedImportances.reduce((a, b) => a + b, 0);
            this.featureImportances = aggregatedImportances.map(v => total > 0 ? Number((v / total).toFixed(4)) : 0);
        }

        predict(X) {
            return X.map(x => this.predictSingle(x).pred);
        }

        predictSingle(x) {
            const votes = this.trees.map(tree => tree.predictSample(x));
            if (this.isClassification) {
                const counts = {};
                votes.forEach(v => counts[v] = (counts[v] || 0) + 1);
                let majority = votes[0];
                let maxCount = 0;
                Object.keys(counts).forEach(v => {
                    if (counts[v] > maxCount) {
                        maxCount = counts[v];
                        majority = Number(v);
                    }
                });
                const probs = {};
                Object.keys(counts).forEach(k => probs[k] = counts[k] / votes.length);
                return { pred: majority, probabilities: probs, confidence: maxCount / votes.length };
            } else {
                const mean = votes.reduce((a, b) => a + b, 0) / votes.length;
                return { pred: Number(mean.toFixed(4)), confidence: 1 };
            }
        }
    }

    // ==========================================
    // 2. LOGISTIC / SOFTMAX & RIDGE REGRESSION
    // ==========================================
    class GeneralizedLinearModel {
        constructor(isClassification = true, numClasses = 2, learningRate = 0.05, epochs = 150, lambdaL2 = 0.01, solver = 'momentum', seed = null) {
            this.isClassification = isClassification;
            this.numClasses = numClasses;
            this.lr = learningRate;
            this.epochs = epochs;
            this.lambdaL2 = lambdaL2;
            this.solver = solver;
            this.seed = seed !== null && seed !== undefined && seed !== '' ? Number(seed) : null;
            this.rng = createPRNG(this.seed);
            this.weights = null; // matrix or vector
            this.bias = null;
        }

        fit(X, y) {
            const n = X.length;
            const p = X[0].length;
            const momentumBeta = this.solver === 'momentum' ? 0.85 : 0.0;

            if (this.isClassification) {
                // Multinomial Softmax or Binary Sigmoid
                this.weights = Array.from({ length: this.numClasses }, () => new Array(p).fill(0));
                this.bias = new Array(this.numClasses).fill(0);
                const velW = Array.from({ length: this.numClasses }, () => new Array(p).fill(0));
                const velB = new Array(this.numClasses).fill(0);

                for (let epoch = 0; epoch < this.epochs; epoch++) {
                    const gradW = Array.from({ length: this.numClasses }, () => new Array(p).fill(0));
                    const gradB = new Array(this.numClasses).fill(0);

                    for (let i = 0; i < n; i++) {
                        const logits = new Array(this.numClasses).fill(0);
                        for (let c = 0; c < this.numClasses; c++) {
                            let dot = this.bias[c];
                            for (let j = 0; j < p; j++) {
                                dot += this.weights[c][j] * X[i][j];
                            }
                            logits[c] = dot;
                        }

                        // Softmax
                        const maxLogit = Math.max(...logits);
                        const exp = logits.map(l => Math.exp(l - maxLogit));
                        const sumExp = exp.reduce((a, b) => a + b, 0);
                        const probs = exp.map(e => e / (sumExp || 1));

                        const trueClass = y[i];
                        for (let c = 0; c < this.numClasses; c++) {
                            const err = probs[c] - (c === trueClass ? 1 : 0);
                            gradB[c] += err;
                            for (let j = 0; j < p; j++) {
                                gradW[c][j] += err * X[i][j];
                            }
                        }
                    }

                    // Update with momentum and regularization
                    for (let c = 0; c < this.numClasses; c++) {
                        const gB = gradB[c] / n;
                        velB[c] = momentumBeta * velB[c] + this.lr * gB;
                        this.bias[c] -= velB[c];

                        for (let j = 0; j < p; j++) {
                            const reg = this.lambdaL2 * this.weights[c][j];
                            const gW = (gradW[c][j] / n) + reg;
                            velW[c][j] = momentumBeta * velW[c][j] + this.lr * gW;
                            this.weights[c][j] -= velW[c][j];
                        }
                    }
                }
            } else {
                // Ridge Linear Regression via Gradient Descent with momentum
                this.weights = new Array(p).fill(0);
                this.bias = 0;
                const velW = new Array(p).fill(0);
                let velB = 0;

                for (let epoch = 0; epoch < this.epochs; epoch++) {
                    let gradB = 0;
                    const gradW = new Array(p).fill(0);

                    for (let i = 0; i < n; i++) {
                        let pred = this.bias;
                        for (let j = 0; j < p; j++) {
                            pred += this.weights[j] * X[i][j];
                        }
                        const err = pred - y[i];
                        gradB += err;
                        for (let j = 0; j < p; j++) {
                            gradW[j] += err * X[i][j];
                        }
                    }

                    const gB = gradB / n;
                    velB = momentumBeta * velB + this.lr * gB;
                    this.bias -= velB;

                    for (let j = 0; j < p; j++) {
                        const reg = this.lambdaL2 * this.weights[j];
                        const gW = (gradW[j] / n) + reg;
                        velW[j] = momentumBeta * velW[j] + this.lr * gW;
                        this.weights[j] -= velW[j];
                    }
                }
            }
        }

        predictSingle(x) {
            const p = x.length;
            if (this.isClassification) {
                const logits = new Array(this.numClasses).fill(0);
                for (let c = 0; c < this.numClasses; c++) {
                    let dot = this.bias[c];
                    for (let j = 0; j < p; j++) {
                        dot += this.weights[c][j] * x[j];
                    }
                    logits[c] = dot;
                }
                const maxLogit = Math.max(...logits);
                const exp = logits.map(l => Math.exp(l - maxLogit));
                const sumExp = exp.reduce((a, b) => a + b, 0);
                const probs = exp.map(e => e / (sumExp || 1));

                let bestClass = 0;
                let maxP = 0;
                const probMap = {};
                probs.forEach((prob, c) => {
                    probMap[c] = Number(prob.toFixed(4));
                    if (prob > maxP) {
                        maxP = prob;
                        bestClass = c;
                    }
                });

                return { pred: bestClass, probabilities: probMap, confidence: Number(maxP.toFixed(4)) };
            } else {
                let pred = this.bias;
                for (let j = 0; j < p; j++) {
                    pred += this.weights[j] * x[j];
                }
                return { pred: Number(pred.toFixed(4)), confidence: 1 };
            }
        }

        predict(X) {
            return X.map(x => this.predictSingle(x).pred);
        }
    }

    // ==========================================
    // 3. k-NEAREST NEIGHBORS (k-NN)
    // ==========================================
    class KNNModel {
        constructor(isClassification = true, k = 5, metric = 'euclidean', weighting = 'uniform', seed = null) {
            this.isClassification = isClassification;
            this.k = k;
            this.metric = metric; // 'euclidean', 'manhattan', 'chebyshev'
            this.weighting = weighting; // 'uniform', 'distance'
            this.seed = seed !== null && seed !== undefined && seed !== '' ? Number(seed) : null;
            this.trainX = [];
            this.trainY = [];
        }

        fit(X, y) {
            this.trainX = X;
            this.trainY = y;
        }

        predictSingle(x) {
            const distances = this.trainX.map((tx, idx) => {
                let d = 0;
                if (this.metric === 'manhattan') {
                    for (let j = 0; j < x.length; j++) {
                        d += Math.abs(x[j] - tx[j]);
                    }
                } else if (this.metric === 'chebyshev') {
                    for (let j = 0; j < x.length; j++) {
                        d = Math.max(d, Math.abs(x[j] - tx[j]));
                    }
                } else {
                    // euclidean
                    let sumSq = 0;
                    for (let j = 0; j < x.length; j++) {
                        const diff = x[j] - tx[j];
                        sumSq += diff * diff;
                    }
                    d = Math.sqrt(sumSq);
                }
                return { dist: d, label: this.trainY[idx] };
            });

            distances.sort((a, b) => a.dist - b.dist);
            const kNearest = distances.slice(0, Math.min(this.k, distances.length));

            if (this.isClassification) {
                const weightedCounts = {};
                let totalWeight = 0;
                kNearest.forEach(n => {
                    const w = this.weighting === 'distance' ? (1.0 / (n.dist + 1e-5)) : 1.0;
                    weightedCounts[n.label] = (weightedCounts[n.label] || 0) + w;
                    totalWeight += w;
                });

                let majority = kNearest[0].label;
                let maxW = 0;
                Object.keys(weightedCounts).forEach(l => {
                    if (weightedCounts[l] > maxW) {
                        maxW = weightedCounts[l];
                        majority = Number(l);
                    }
                });

                const probs = {};
                Object.keys(weightedCounts).forEach(k => {
                    probs[k] = Number((weightedCounts[k] / (totalWeight || 1)).toFixed(4));
                });
                return {
                    pred: majority,
                    probabilities: probs,
                    confidence: Number((maxW / (totalWeight || 1)).toFixed(4))
                };
            } else {
                let weightedSum = 0;
                let totalWeight = 0;
                kNearest.forEach(n => {
                    const w = this.weighting === 'distance' ? (1.0 / (n.dist + 1e-5)) : 1.0;
                    weightedSum += n.label * w;
                    totalWeight += w;
                });
                const mean = weightedSum / (totalWeight || 1);
                return { pred: Number(mean.toFixed(4)), confidence: 1 };
            }
        }

        predict(X) {
            return X.map(x => this.predictSingle(x).pred);
        }
    }

    // ==========================================
    // 4. GAUSSIAN NAIVE BAYES
    // ==========================================
    class GaussianNaiveBayes {
        constructor(varSmoothing = 1e-4, priorMode = 'empirical', seed = null) {
            this.varSmoothing = varSmoothing; // 1e-9, 1e-4, 1e-2, 0.1
            this.priorMode = priorMode; // 'empirical', 'uniform', 'laplace'
            this.seed = seed !== null && seed !== undefined && seed !== '' ? Number(seed) : null;
            this.classes = [];
            this.priors = {};
            this.means = {};
            this.variances = {};
        }

        fit(X, y) {
            this.classes = [...new Set(y)];
            const n = X.length;
            const p = X[0].length;
            const numK = this.classes.length;

            this.classes.forEach(c => {
                const classX = X.filter((_, idx) => y[idx] === c);
                const count = classX.length;

                if (this.priorMode === 'uniform') {
                    this.priors[c] = 1.0 / (numK || 1);
                } else if (this.priorMode === 'laplace') {
                    this.priors[c] = (count + 1) / (n + numK);
                } else {
                    // empirical
                    this.priors[c] = count / (n || 1);
                }

                this.means[c] = new Array(p).fill(0);
                this.variances[c] = new Array(p).fill(this.varSmoothing);

                for (let j = 0; j < p; j++) {
                    const vals = classX.map(r => r[j]);
                    const mean = vals.reduce((a, b) => a + b, 0) / (vals.length || 1);
                    this.means[c][j] = mean;
                    const variance = vals.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / (vals.length || 1);
                    this.variances[c][j] = Math.max(variance, this.varSmoothing);
                }
            });
        }

        predictSingle(x) {
            const logProbs = {};
            const p = x.length;

            this.classes.forEach(c => {
                let logP = Math.log(this.priors[c] || 1e-6);
                for (let j = 0; j < p; j++) {
                    const mean = this.means[c][j];
                    const v = this.variances[c][j];
                    const diff = x[j] - mean;
                    logP += -0.5 * Math.log(2 * Math.PI * v) - (diff * diff) / (2 * v);
                }
                logProbs[c] = logP;
            });

            const maxLogP = Math.max(...Object.values(logProbs));
            const exps = {};
            let sumExp = 0;
            Object.keys(logProbs).forEach(c => {
                const exp = Math.exp(logProbs[c] - maxLogP);
                exps[c] = exp;
                sumExp += exp;
            });

            let bestClass = this.classes[0];
            let maxProb = 0;
            const probMap = {};
            Object.keys(exps).forEach(c => {
                const prob = exps[c] / (sumExp || 1);
                probMap[c] = Number(prob.toFixed(4));
                if (prob > maxProb) {
                    maxProb = prob;
                    bestClass = Number(c);
                }
            });

            return { pred: bestClass, probabilities: probMap, confidence: Number(maxProb.toFixed(4)) };
        }

        predict(X) {
            return X.map(x => this.predictSingle(x).pred);
        }
    }

    // ==========================================
    // 5. TENSORFLOW.JS / NATIVE DEEP NEURAL NETWORK
    // ==========================================
    class DeepNeuralNetwork {
        constructor(isClassification = true, numClasses = 2, config = {}) {
            this.isClassification = isClassification;
            this.numClasses = numClasses;
            this.hiddenLayers = config.hiddenLayers || [64, 32];
            this.activation = config.activation || 'relu';
            this.learningRate = config.learningRate || 0.005;
            this.epochs = config.epochs || 50;
            this.batchSize = config.batchSize || 16;
            this.dropoutRate = config.dropoutRate !== undefined ? config.dropoutRate : 0.1;
            this.optimizerName = config.optimizer || 'adam';
            this.l2Reg = config.l2Reg !== undefined ? config.l2Reg : 0.0001;
            this.seed = config.seed !== null && config.seed !== undefined && config.seed !== '' ? Number(config.seed) : null;
            this.rng = createPRNG(this.seed);
            this.tfModel = null;
            this.fallbackGLM = null;
            this.history = { loss: [], valLoss: [], accuracy: [], valAccuracy: [] };
        }

        async fit(trainX, trainY, testX = null, testY = null, onEpochCallback = null) {
            const hasTF = typeof window.tf !== 'undefined' && window.tf !== null;
            const inputDim = trainX[0].length;

            if (hasTF) {
                // Dispose existing model instance if retraining to prevent GPU memory leak
                if (this.tfModel) {
                    try { this.tfModel.dispose(); } catch (e) { /* ignore */ }
                    this.tfModel = null;
                }

                let xTensor = null;
                let yTensor = null;
                let valXTensor = null;
                let valYTensor = null;

                try {
                    const tf = window.tf;
                    const model = tf.sequential();

                    // Optional L2 kernel regularizer
                    const kernelReg = (this.l2Reg && this.l2Reg > 0 && typeof tf.regularizers !== 'undefined' && tf.regularizers.l2)
                        ? tf.regularizers.l2({ l2: this.l2Reg })
                        : null;

                    // Seed-based weight initializer for reproducibility
                    const kernelInit = (this.seed !== null && typeof tf.initializers !== 'undefined' && tf.initializers.glorotUniform)
                        ? tf.initializers.glorotUniform({ seed: this.seed })
                        : 'glorotUniform';

                    // Input + First Hidden Layer
                    const firstDense = {
                        inputShape: [inputDim],
                        units: this.hiddenLayers[0] || 64,
                        activation: this.activation,
                        kernelInitializer: kernelInit
                    };
                    if (kernelReg) firstDense.kernelRegularizer = kernelReg;
                    model.add(tf.layers.dense(firstDense));

                    if (this.dropoutRate > 0) {
                        model.add(tf.layers.dropout({ rate: this.dropoutRate, seed: this.seed }));
                    }

                    // Subsequent Hidden Layers
                    for (let l = 1; l < this.hiddenLayers.length; l++) {
                        const units = this.hiddenLayers[l];
                        if (!units || units <= 0) continue;
                        const nextDense = {
                            units: units,
                            activation: this.activation,
                            kernelInitializer: kernelInit
                        };
                        if (kernelReg) nextDense.kernelRegularizer = kernelReg;
                        model.add(tf.layers.dense(nextDense));
                        if (this.dropoutRate > 0) {
                            model.add(tf.layers.dropout({ rate: this.dropoutRate, seed: this.seed ? this.seed + l : null }));
                        }
                    }

                    // Output Layer
                    if (this.isClassification) {
                        model.add(tf.layers.dense({
                            units: this.numClasses,
                            activation: 'softmax',
                            kernelInitializer: kernelInit
                        }));
                    } else {
                        model.add(tf.layers.dense({ units: 1, kernelInitializer: kernelInit }));
                    }

                    // Optimizer
                    let opt;
                    if (this.optimizerName === 'sgd') {
                        opt = tf.train.sgd(this.learningRate);
                    } else if (this.optimizerName === 'rmsprop') {
                        opt = tf.train.rmsprop(this.learningRate);
                    } else {
                        opt = tf.train.adam(this.learningRate);
                    }

                    model.compile({
                        optimizer: opt,
                        loss: this.isClassification ? 'sparseCategoricalCrossentropy' : 'meanSquaredError',
                        metrics: this.isClassification ? ['accuracy'] : ['mse']
                    });

                    // Seed-reproducible training data shuffling using deterministic PRNG
                    const indices = trainX.map((_, idx) => idx);
                    for (let i = indices.length - 1; i > 0; i--) {
                        const j = Math.floor(this.rng() * (i + 1));
                        const tmp = indices[i];
                        indices[i] = indices[j];
                        indices[j] = tmp;
                    }
                    const shuffledTrainX = indices.map(idx => trainX[idx]);
                    const shuffledTrainY = indices.map(idx => trainY[idx]);

                    // Prepare Tensors
                    xTensor = tf.tensor2d(shuffledTrainX, [shuffledTrainX.length, inputDim]);
                    yTensor = this.isClassification ?
                        tf.tensor1d(shuffledTrainY, 'int32') :
                        tf.tensor2d(shuffledTrainY.map(v => [v]), [shuffledTrainY.length, 1]);

                    let validationData = null;
                    if (testX && testY && testX.length > 0) {
                        valXTensor = tf.tensor2d(testX, [testX.length, inputDim]);
                        valYTensor = this.isClassification ?
                            tf.tensor1d(testY, 'int32') :
                            tf.tensor2d(testY.map(v => [v]), [testY.length, 1]);
                        validationData = [valXTensor, valYTensor];
                    }

                    const history = { loss: [], valLoss: [], accuracy: [], valAccuracy: [] };

                    await model.fit(xTensor, yTensor, {
                        epochs: this.epochs,
                        batchSize: Math.min(this.batchSize, trainX.length),
                        validationData: validationData,
                        shuffle: false, // deterministic: already seeded-shuffled above
                        callbacks: {
                            onEpochEnd: (epoch, logs) => {
                                history.loss.push(Number(logs.loss.toFixed(4)));
                                if (logs.val_loss !== undefined) history.valLoss.push(Number(logs.val_loss.toFixed(4)));
                                if (logs.acc !== undefined) history.accuracy.push(Number(logs.acc.toFixed(4)));
                                if (logs.val_acc !== undefined) history.valAccuracy.push(Number(logs.val_acc.toFixed(4)));

                                if (onEpochCallback) {
                                    onEpochCallback(epoch + 1, this.epochs, logs);
                                }
                            }
                        }
                    });

                    this.tfModel = model;
                    this.history = history;
                    return;
                } catch (e) {
                    console.warn("TensorFlow.js training exception, using native fallback:", e);
                } finally {
                    // Guaranteed cleanup of intermediate tensors
                    if (xTensor) { try { xTensor.dispose(); } catch (e) {} }
                    if (yTensor) { try { yTensor.dispose(); } catch (e) {} }
                    if (valXTensor) { try { valXTensor.dispose(); } catch (e) {} }
                    if (valYTensor) { try { valYTensor.dispose(); } catch (e) {} }
                }
            }

            // Fallback to Native Logistic/Softmax/Ridge
            this.fallbackGLM = new GeneralizedLinearModel(
                this.isClassification,
                this.numClasses,
                this.learningRate,
                this.epochs,
                this.l2Reg,
                'momentum'
            );
            this.fallbackGLM.fit(trainX, trainY);
            this.history = { loss: [], valLoss: [], accuracy: [], valAccuracy: [] };
        }

        predictSingle(x) {
            if (this.tfModel) {
                const tf = window.tf;
                const tensor = tf.tensor2d([x], [1, x.length]);
                const output = this.tfModel.predict(tensor);
                const data = output.dataSync();
                tensor.dispose();
                output.dispose();

                if (this.isClassification) {
                    let bestClass = 0;
                    let maxP = 0;
                    const probMap = {};
                    for (let c = 0; c < this.numClasses; c++) {
                        const prob = data[c];
                        probMap[c] = Number(prob.toFixed(4));
                        if (prob > maxP) {
                            maxP = prob;
                            bestClass = c;
                        }
                    }
                    return { pred: bestClass, probabilities: probMap, confidence: Number(maxP.toFixed(4)) };
                } else {
                    return { pred: Number(data[0].toFixed(4)), confidence: 1 };
                }
            } else if (this.fallbackGLM) {
                return this.fallbackGLM.predictSingle(x);
            }
            throw new Error("Model is not trained yet.");
        }

        predict(X) {
            return X.map(x => this.predictSingle(x).pred);
        }
    }

    // ==========================================
    // 6. ENSEMBLE SUPER LEARNER (VOTING & BLENDING)
    // ==========================================
    class EnsembleModel {
        constructor(isClassification, numClasses = 2, config = {}) {
            this.isClassification = isClassification;
            this.numClasses = numClasses;
            this.voting = config.voting || 'soft'; // 'soft', 'weighted', 'hard'
            this.selectedBaseIds = config.baseModels && config.baseModels.length > 0
                ? config.baseModels
                : ['nn', 'rf', 'glm'];
            this.seed = config.seed !== null && config.seed !== undefined && config.seed !== '' ? Number(config.seed) : null;
            this.rng = createPRNG(this.seed);
            this.models = [];
            this.weights = [];
            this.baseMetrics = [];
            this.userHyperparams = config;
        }

        async fit(trainX, trainY, testX = null, testY = null, onProgress = null) {
            this.models = [];
            this.weights = [];
            this.baseMetrics = [];

            const totalBases = this.selectedBaseIds.length;
            const capacity = this.userHyperparams.capacity || 'balanced';

            for (let idx = 0; idx < totalBases; idx++) {
                const mId = this.selectedBaseIds[idx];
                if (!this.isClassification && mId === 'nb') {
                    continue; // Naive Bayes is classification only
                }

                if (onProgress) {
                    onProgress(idx + 1, totalBases, {
                        stage: `Training Ensemble Base Model ${idx + 1}/${totalBases}: ${mId.toUpperCase()}...`,
                        baseId: mId
                    });
                }

                const subSeed = this.seed !== null ? (this.seed + (idx + 1) * 1007) : null;
                let subModel = null;
                if (mId === 'nn') {
                    const epochs = capacity === 'fast' ? 20 : (capacity === 'deep' ? 60 : (this.userHyperparams.nnEpochs || 35));
                    const hidden = capacity === 'fast' ? [32, 16] : (capacity === 'deep' ? [64, 48, 24] : [48, 24]);
                    const nnParams = {
                        epochs: epochs,
                        learningRate: this.userHyperparams.nnLr || 0.005,
                        hiddenLayers: hidden,
                        dropoutRate: 0.1,
                        activation: 'relu',
                        optimizer: 'adam',
                        batchSize: 16,
                        l2Reg: 0.0001,
                        seed: subSeed
                    };
                    subModel = new DeepNeuralNetwork(this.isClassification, this.numClasses, nnParams);
                    await subModel.fit(trainX, trainY, testX, testY, (ep, tot, logs) => {
                        if (onProgress) {
                            onProgress(idx + 1, totalBases, {
                                stage: `Base ${idx + 1}/${totalBases} (NN Epoch ${ep}/${tot} - Loss: ${logs.loss.toFixed(4)})`,
                                loss: logs.loss,
                                val_loss: logs.val_loss
                            });
                        }
                    });
                } else if (mId === 'rf') {
                    const trees = capacity === 'fast' ? 15 : (capacity === 'deep' ? 50 : (this.userHyperparams.rfTrees || 30));
                    const depth = capacity === 'fast' ? 4 : (capacity === 'deep' ? 8 : (this.userHyperparams.rfDepth || 6));
                    subModel = new RandomForest(this.isClassification, trees, depth, 2, 'sqrt', subSeed);
                    subModel.fit(trainX, trainY);
                } else if (mId === 'glm') {
                    const epochs = capacity === 'fast' ? 80 : (capacity === 'deep' ? 250 : 150);
                    subModel = new GeneralizedLinearModel(this.isClassification, this.numClasses, 0.05, epochs, 0.01, 'momentum', subSeed);
                    subModel.fit(trainX, trainY);
                } else if (mId === 'knn') {
                    const kVal = capacity === 'fast' ? 3 : (capacity === 'deep' ? 7 : (this.userHyperparams.knnK || 5));
                    subModel = new KNNModel(this.isClassification, kVal, 'euclidean', 'distance', subSeed);
                    subModel.fit(trainX, trainY);
                } else if (mId === 'nb' && this.isClassification) {
                    subModel = new GaussianNaiveBayes(1e-4, 'empirical', subSeed);
                    subModel.fit(trainX, trainY);
                }

                if (subModel) {
                    this.models.push({ id: mId, model: subModel });
                }
            }

            // Determine weights strictly on an internal training fold partition (zero leakage into test set)
            let valX = [];
            let valY = [];
            if (trainX.length >= 10) {
                const valSize = Math.max(2, Math.floor(trainX.length * 0.2));
                valX = trainX.slice(-valSize);
                valY = trainY.slice(-valSize);
            } else {
                valX = trainX;
                valY = trainY;
            }

            const power = this.userHyperparams.power || 2;
            let rawWeights = [];

            if (this.voting === 'weighted' && valX.length > 0) {
                for (let item of this.models) {
                    const preds = item.model.predict(valX);
                    if (this.isClassification) {
                        let correct = 0;
                        for (let i = 0; i < valY.length; i++) {
                            if (preds[i] === valY[i]) correct++;
                        }
                        const acc = correct / valY.length;
                        const w = Math.pow(Math.max(0.05, acc), power);
                        rawWeights.push(w);
                        this.baseMetrics.push({ id: item.id, accuracy: acc, weight: w });
                    } else {
                        let sumSq = 0;
                        for (let i = 0; i < valY.length; i++) {
                            sumSq += Math.pow(valY[i] - preds[i], 2);
                        }
                        const mse = sumSq / valY.length;
                        const w = Math.pow(1.0 / (mse + 0.001), power / 2);
                        rawWeights.push(w);
                        this.baseMetrics.push({ id: item.id, mse, weight: w });
                    }
                }
            } else {
                rawWeights = new Array(this.models.length).fill(1.0);
                this.models.forEach(item => {
                    this.baseMetrics.push({ id: item.id, weight: 1.0 });
                });
            }

            const totalW = rawWeights.reduce((a, b) => a + b, 0) || 1.0;
            this.weights = rawWeights.map(w => w / totalW);
            this.baseMetrics.forEach((m, i) => {
                m.normalizedWeight = Number(this.weights[i].toFixed(3));
            });
        }

        predictSingle(x) {
            if (!this.models || this.models.length === 0) {
                throw new Error("Ensemble model is not trained yet.");
            }

            if (this.isClassification) {
                if (this.voting === 'hard') {
                    const votes = new Array(this.numClasses).fill(0);
                    this.models.forEach((item, i) => {
                        const res = item.model.predictSingle(x);
                        const predClass = res.pred;
                        if (predClass >= 0 && predClass < this.numClasses) {
                            votes[predClass] += this.weights[i];
                        }
                    });

                    let bestClass = 0;
                    let maxVotes = -1;
                    votes.forEach((v, c) => {
                        if (v > maxVotes) {
                            maxVotes = v;
                            bestClass = c;
                        }
                    });

                    const totalVotes = votes.reduce((a, b) => a + b, 0) || 1;
                    const probMap = {};
                    votes.forEach((v, c) => {
                        probMap[c] = Number((v / totalVotes).toFixed(4));
                    });

                    return {
                        pred: bestClass,
                        probabilities: probMap,
                        confidence: Number((maxVotes / totalVotes).toFixed(4)),
                        ensembleDetails: this.baseMetrics
                    };
                } else {
                    // Soft voting (weighted probability average)
                    const combinedProbs = new Array(this.numClasses).fill(0);

                    this.models.forEach((item, i) => {
                        const res = item.model.predictSingle(x);
                        const w = this.weights[i];

                        if (res.probabilities) {
                            for (let c = 0; c < this.numClasses; c++) {
                                const p = res.probabilities[c] !== undefined ? res.probabilities[c] : (res.pred === c ? 1 : 0);
                                combinedProbs[c] += w * p;
                            }
                        } else {
                            combinedProbs[res.pred] += w;
                        }
                    });

                    let bestClass = 0;
                    let maxP = -1;
                    const probMap = {};

                    combinedProbs.forEach((p, c) => {
                        const clampedP = Number(Math.max(0, Math.min(1, p)).toFixed(4));
                        probMap[c] = clampedP;
                        if (clampedP > maxP) {
                            maxP = clampedP;
                            bestClass = c;
                        }
                    });

                    return {
                        pred: bestClass,
                        probabilities: probMap,
                        confidence: Number(maxP.toFixed(4)),
                        ensembleDetails: this.baseMetrics
                    };
                }
            } else {
                // Regression: weighted average of continuous outputs
                let weightedSum = 0;
                const subPredictions = [];

                this.models.forEach((item, i) => {
                    const res = item.model.predictSingle(x);
                    weightedSum += this.weights[i] * res.pred;
                    subPredictions.push({ id: item.id, pred: res.pred, weight: this.weights[i] });
                });

                return {
                    pred: Number(weightedSum.toFixed(4)),
                    confidence: 1.0,
                    subPredictions,
                    ensembleDetails: this.baseMetrics
                };
            }
        }

        predict(X) {
            return X.map(x => this.predictSingle(x).pred);
        }
    }

    // ==========================================
    // 7. MODEL MANAGER & EVALUATION METRICS
    // ==========================================
    const ModelEngine = {
        SUPPORTED_MODELS: [
            { id: 'ensemble', name: 'Ensemble Meta-Learner (Voting & Blending)', type: 'both', desc: 'Clinical ensemble combining Deep Neural Networks, Random Forests, Logistic Regression, and k-NN with soft probability blending.' },
            { id: 'nn', name: 'Deep Neural Network (MLP / TF.js)', type: 'both', desc: 'Multi-layer perceptron with backpropagation, dense hidden layers, dropout & Adam optimizer.' },
            { id: 'rf', name: 'Random Forest (Ensemble Trees)', type: 'both', desc: 'Ensemble of bootstrap decision trees with out-of-bag diversity and feature importance ranking.' },
            { id: 'glm', name: 'Logistic / Ridge Regression', type: 'both', desc: 'L2-regularized linear/softmax model providing interpretable odds ratios and coefficients.' },
            { id: 'knn', name: 'k-Nearest Neighbors (k-NN)', type: 'both', desc: 'Non-parametric distance-based instance retrieval model for local pattern recognition.' },
            { id: 'nb', name: 'Gaussian Naive Bayes', type: 'classification', desc: 'Probabilistic baseline utilizing Gaussian likelihood distribution per feature.' }
        ],

        /**
         * Create an instance of a model based on user selection
         */
        createModel(modelId, isClassification, numClasses = 2, userHyperparams = {}) {
            let model;
            const seed = userHyperparams.seed !== undefined && userHyperparams.seed !== null ? userHyperparams.seed : null;
            if (modelId === 'ensemble') {
                model = new EnsembleModel(isClassification, numClasses, userHyperparams);
            } else if (modelId === 'nn') {
                model = new DeepNeuralNetwork(isClassification, numClasses, userHyperparams);
            } else if (modelId === 'rf') {
                const trees = userHyperparams.numTrees || 30;
                const depth = userHyperparams.maxDepth || 6;
                const minSplit = userHyperparams.minSamplesSplit || 2;
                const featRatio = userHyperparams.featureRatio || 'sqrt';
                model = new RandomForest(isClassification, trees, depth, minSplit, featRatio, seed);
            } else if (modelId === 'glm') {
                model = new GeneralizedLinearModel(
                    isClassification,
                    numClasses,
                    userHyperparams.lr || 0.05,
                    userHyperparams.epochs || 150,
                    userHyperparams.lambdaL2 !== undefined ? userHyperparams.lambdaL2 : 0.01,
                    userHyperparams.solver || 'momentum',
                    seed
                );
            } else if (modelId === 'knn') {
                model = new KNNModel(
                    isClassification,
                    userHyperparams.k || 5,
                    userHyperparams.metric || 'euclidean',
                    userHyperparams.weighting || 'uniform',
                    seed
                );
            } else if (modelId === 'nb') {
                if (!isClassification) throw new Error("Naive Bayes is only supported for classification tasks.");
                model = new GaussianNaiveBayes(
                    userHyperparams.varSmoothing !== undefined ? userHyperparams.varSmoothing : 1e-4,
                    userHyperparams.priorMode || 'empirical',
                    seed
                );
            } else {
                throw new Error(`Unknown model ID: ${modelId}`);
            }
            if (model && seed !== null && seed !== undefined) {
                model.seed = seed;
            }
            return model;
        },

        /**
         * Evaluate model performance on test set
         */
        evaluate(model, testX, testY, isClassification, targetClasses = []) {
            if (!testX || testX.length === 0) {
                return null;
            }

            const preds = model.predict(testX);

            if (isClassification) {
                const numClasses = targetClasses.length || [...new Set(testY)].length;
                let correct = 0;
                const total = testX.length;

                // Confusion Matrix [actual][pred]
                const confMatrix = Array.from({ length: numClasses }, () => new Array(numClasses).fill(0));

                for (let i = 0; i < total; i++) {
                    const actual = testY[i];
                    const pred = preds[i];
                    if (actual === pred) correct++;
                    if (actual < numClasses && pred < numClasses) {
                        confMatrix[actual][pred]++;
                    }
                }

                const accuracy = Number((correct / total).toFixed(4));

                // Per-class metrics
                const classMetrics = targetClasses.map((clsName, cIdx) => {
                    let tp = confMatrix[cIdx][cIdx];
                    let fp = 0;
                    let fn = 0;
                    let support = 0;

                    for (let r = 0; r < numClasses; r++) {
                        support += confMatrix[cIdx][r];
                        if (r !== cIdx) {
                            fn += confMatrix[cIdx][r];
                            fp += confMatrix[r][cIdx];
                        }
                    }

                    const precision = (tp + fp) > 0 ? Number((tp / (tp + fp)).toFixed(4)) : 0;
                    const recall = (tp + fn) > 0 ? Number((tp / (tp + fn)).toFixed(4)) : 0;
                    const f1 = (precision + recall) > 0 ? Number(((2 * precision * recall) / (precision + recall)).toFixed(4)) : 0;

                    return {
                        className: clsName,
                        precision,
                        recall,
                        f1,
                        support
                    };
                });

                // Macro-averaged metrics
                const macroPrecision = Number((classMetrics.reduce((a, b) => a + b.precision, 0) / numClasses).toFixed(4));
                const macroRecall = Number((classMetrics.reduce((a, b) => a + b.recall, 0) / numClasses).toFixed(4));
                const macroF1 = Number((classMetrics.reduce((a, b) => a + b.f1, 0) / numClasses).toFixed(4));

                // ROC-AUC for binary classification
                let rocAuc = null;
                if (numClasses === 2) {
                    try {
                        const probs = testX.map(x => {
                            const res = model.predictSingle(x);
                            return res.probabilities && res.probabilities[1] !== undefined ? res.probabilities[1] : (res.pred === 1 ? 1 : 0);
                        });
                        rocAuc = this._calculateBinaryAUC(testY, probs);
                    } catch (e) {
                        rocAuc = null;
                    }
                }

                return {
                    isClassification: true,
                    totalSamples: total,
                    accuracy,
                    macroPrecision,
                    macroRecall,
                    macroF1,
                    rocAuc,
                    confMatrix,
                    classMetrics,
                    actuals: testY,
                    predictions: preds,
                    seed: model.seed !== undefined ? model.seed : null
                };
            } else {
                // Regression Metrics: R2, MSE, RMSE, MAE
                const total = testX.length;
                let sumSqErr = 0;
                let sumAbsErr = 0;
                const meanActual = testY.reduce((a, b) => a + b, 0) / total;
                let totalSq = 0;

                for (let i = 0; i < total; i++) {
                    const actual = testY[i];
                    const pred = preds[i];
                    const err = actual - pred;
                    sumSqErr += err * err;
                    sumAbsErr += Math.abs(err);
                    totalSq += Math.pow(actual - meanActual, 2);
                }

                const mse = Number((sumSqErr / total).toFixed(4));
                const rmse = Number((Math.sqrt(mse)).toFixed(4));
                const mae = Number((sumAbsErr / total).toFixed(4));
                const r2 = totalSq > 0 ? Number((1 - (sumSqErr / totalSq)).toFixed(4)) : 0;

                // 2x2 Clinical Dichotomization / Concordance Confusion Matrix for Continuous Target
                const sortedActuals = [...testY].sort((a, b) => a - b);
                const medianThreshold = sortedActuals.length % 2 === 0
                    ? Number(((sortedActuals[sortedActuals.length / 2 - 1] + sortedActuals[sortedActuals.length / 2]) / 2).toFixed(2))
                    : Number(sortedActuals[Math.floor(sortedActuals.length / 2)].toFixed(2));
                const threshold = medianThreshold;

                // 2x2 table: Row 0 = Actual < Thresh, Row 1 = Actual >= Thresh
                // Col 0 = Pred < Thresh, Col 1 = Pred >= Thresh
                const confMatrix2x2 = [
                    [0, 0],
                    [0, 0]
                ];
                let hits = 0;
                for (let i = 0; i < total; i++) {
                    const actBin = testY[i] >= threshold ? 1 : 0;
                    const predBin = preds[i] >= threshold ? 1 : 0;
                    confMatrix2x2[actBin][predBin]++;
                    if (actBin === predBin) hits++;
                }
                const thresholdAccuracy = Number((hits / total).toFixed(4));
                const tp = confMatrix2x2[1][1];
                const fn = confMatrix2x2[1][0];
                const fp = confMatrix2x2[0][1];
                const tn = confMatrix2x2[0][0];
                const sensitivity = (tp + fn) > 0 ? Number((tp / (tp + fn)).toFixed(4)) : 0;
                const specificity = (tn + fp) > 0 ? Number((tn / (tn + fp)).toFixed(4)) : 0;

                return {
                    isClassification: false,
                    totalSamples: total,
                    r2,
                    mse,
                    rmse,
                    mae,
                    actuals: testY,
                    predictions: preds,
                    confMatrix: confMatrix2x2,
                    threshold: threshold,
                    thresholdAccuracy: thresholdAccuracy,
                    sensitivity: sensitivity,
                    specificity: specificity,
                    seed: model.seed !== undefined ? model.seed : null
                };
            }
        },

        evaluateModel(model, testX, testY, isClassificationOrClasses = true, targetClassesOrStats = []) {
            let isClassification = true;
            let targetClasses = [];
            if (typeof isClassificationOrClasses === 'boolean') {
                isClassification = isClassificationOrClasses;
                targetClasses = Array.isArray(targetClassesOrStats) ? targetClassesOrStats : [];
            } else if (Array.isArray(isClassificationOrClasses)) {
                targetClasses = isClassificationOrClasses;
                isClassification = targetClasses.length > 0;
            } else if (model && model.isClassification !== undefined) {
                isClassification = Boolean(model.isClassification);
                targetClasses = Array.isArray(targetClassesOrStats) ? targetClassesOrStats : [];
            }
            return this.evaluate(model, testX, testY, isClassification, targetClasses);
        },

        _calculateBinaryAUC(yTrue, yScores) {
            // Rank-based Mann-Whitney U test calculation for ROC AUC with mid-rank tie handling
            const n = yTrue.length;
            if (n === 0) return null;

            const items = yTrue.map((val, idx) => ({ y: val, score: yScores[idx] }));
            items.sort((a, b) => a.score - b.score);

            let nPos = 0;
            let nNeg = 0;
            items.forEach(it => {
                if (it.y === 1) nPos++;
                else nNeg++;
            });

            // ROC-AUC is undefined when test set lacks either positive or negative class
            if (nPos === 0 || nNeg === 0) return null;

            // Assign mid-ranks for tied scores
            const ranks = new Array(n);
            let i = 0;
            while (i < n) {
                let j = i;
                while (j < n - 1 && items[j + 1].score === items[i].score) {
                    j++;
                }
                // Items from i to j have tied scores; rank is 1-based average: ( (i+1) + ... + (j+1) ) / (j - i + 1)
                const avgRank = (i + 1 + j + 1) / 2;
                for (let k = i; k <= j; k++) {
                    ranks[k] = avgRank;
                }
                i = j + 1;
            }

            let rankSumPos = 0;
            items.forEach((it, idx) => {
                if (it.y === 1) {
                    rankSumPos += ranks[idx];
                }
            });

            const u = rankSumPos - (nPos * (nPos + 1)) / 2;
            const auc = u / (nPos * nNeg);
            return Number(Math.max(0, Math.min(1, auc)).toFixed(4));
        }
    };

    window.ModelEngine = ModelEngine;
    window.ModelFactory = ModelEngine;
    window.RandomForest = RandomForest;
    window.KNNModel = KNNModel;
    window.GeneralizedLinearModel = GeneralizedLinearModel;
    window.DeepNeuralNetwork = DeepNeuralNetwork;
    window.GaussianNaiveBayes = GaussianNaiveBayes;
    window.EnsembleModel = EnsembleModel;
})(window);
