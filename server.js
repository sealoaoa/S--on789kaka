const WebSocket = require('ws');
const express = require('express');
const cors = require('cors');
const math = require('mathjs');
const { LogisticRegression } = require('ml-logistic-regression');
const stats = require('simple-statistics');

// ==================== LỚP QUẢN LÝ LỊCH SỬ ====================
class HistoryManager {
    constructor(maxSize = 100) {
        this.maxSize = maxSize;
        this.data = {
            tx: [],
            md5: []
        };
    }

    // Thêm phiên mới, tránh trùng sid
    addSession(table, session) {
        const arr = this.data[table];
        if (arr.some(s => s.sid === session.sid)) return false;
        arr.push(session);
        arr.sort((a, b) => a.sid - b.sid);
        if (arr.length > this.maxSize) {
            this.data[table] = arr.slice(-this.maxSize);
        }
        return true;
    }

    // Thêm nhiều phiên từ gameData.htr
    addSessionsFromGameData(table, gameData) {
        if (!gameData || !gameData.htr) return 0;
        let added = 0;
        for (const sess of gameData.htr) {
            const tong = sess.d1 + sess.d2 + sess.d3;
            const ketQua = tong >= 11 ? 'tài' : 'xỉu';
            const session = {
                sid: sess.sid,
                d1: sess.d1,
                d2: sess.d2,
                d3: sess.d3,
                tong: tong,
                ket_qua: ketQua,
                thoi_gian: new Date().toISOString()
            };
            if (this.addSession(table, session)) added++;
        }
        return added;
    }

    getHistory(table, limit = 100) {
        const arr = this.data[table] || [];
        return arr.slice(-limit);
    }

    count(table) {
        return (this.data[table] || []).length;
    }

    getRecentResults(table, n) {
        const arr = this.data[table] || [];
        return arr.slice(-n).map(s => s.ket_qua === 'tài' ? 1 : 0);
    }

    getRecentTong(table, n) {
        const arr = this.data[table] || [];
        return arr.slice(-n).map(s => s.tong);
    }
}

// ==================== LỚP DỰ ĐOÁN ENSEMBLE ====================
class Predictor {
    constructor() {
        this.modelWeights = {
            markov1: { weight: 1, acc: [] },
            markov2: { weight: 1, acc: [] },
            markov3: { weight: 1, acc: [] },
            logistic: { weight: 1, acc: [] },
            naiveBayes: { weight: 1, acc: [] }
        };
        this.logisticModel = null;
    }

    markovProbability(history, order) {
        if (history.length < order + 1) return 0.5;
        const results = history;
        let countPattern = 0;
        let countNext1 = 0;
        for (let i = 0; i <= results.length - order - 1; i++) {
            let match = true;
            for (let j = 0; j < order; j++) {
                if (results[i + j] !== results[results.length - order + j]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                countPattern++;
                if (results[i + order] === 1) countNext1++;
            }
        }
        if (countPattern === 0) return 0.5;
        return countNext1 / countPattern;
    }

    extractFeatures(history) {
        const n = history.length;
        const results = history.map(s => s.ket_qua === 'tài' ? 1 : 0);
        const tongs = history.map(s => s.tong);
        
        const recent5 = results.slice(-5);
        const rate5 = recent5.length ? recent5.reduce((a,b)=>a+b,0)/recent5.length : 0.5;
        const recent10 = results.slice(-10);
        const rate10 = recent10.length ? recent10.reduce((a,b)=>a+b,0)/recent10.length : 0.5;
        const recent20 = results.slice(-20);
        const rate20 = recent20.length ? recent20.reduce((a,b)=>a+b,0)/recent20.length : 0.5;

        const last3 = results.slice(-3).concat([0,0,0]).slice(0,3);

        const tong5 = tongs.slice(-5);
        const meanTong = tong5.length ? stats.mean(tong5) : 10.5;
        const stdTong = tong5.length > 1 ? stats.standardDeviation(tong5) : 2.5;

        return [rate5, rate10, rate20, ...last3, meanTong, stdTong];
    }

    trainLogistic(history) {
        if (history.length < 20) return null;
        const X = [];
        const y = [];
        for (let i = 10; i < history.length; i++) {
            const recent = history.slice(0, i);
            const features = this.extractFeatures(recent);
            X.push(features);
            y.push(history[i].ket_qua === 'tài' ? 1 : 0);
        }
        if (X.length < 5) return null;
        const model = new LogisticRegression({ numSteps: 100, learningRate: 0.1 });
        model.train(X, y);
        return model;
    }

    naiveBayes(history) {
        if (history.length === 0) return 0.5;
        const results = history.map(s => s.ket_qua === 'tài' ? 1 : 0);
        return stats.mean(results);
    }

    predict(table, historyManager) {
        const history = historyManager.getHistory(table, 100);
        if (history.length < 10) {
            return { error: "Chưa đủ dữ liệu (cần ít nhất 10 phiên)" };
        }

        const recentResults = history.map(s => s.ket_qua === 'tài' ? 1 : 0);

        // Markov
        const probMarkov1 = this.markovProbability(recentResults, 1);
        const probMarkov2 = this.markovProbability(recentResults, 2);
        const probMarkov3 = this.markovProbability(recentResults, 3);

        // Logistic
        let probLogistic = 0.5;
        if (history.length >= 20) {
            if (!this.logisticModel) {
                this.logisticModel = this.trainLogistic(history);
            }
            if (this.logisticModel) {
                const features = this.extractFeatures(history);
                probLogistic = this.logisticModel.predict([features])[0];
            }
        }

        // Naive Bayes
        const probNB = this.naiveBayes(history);

        // Ensemble có trọng số
        const probs = [
            { name: 'markov1', prob: probMarkov1, weight: this.modelWeights.markov1.weight },
            { name: 'markov2', prob: probMarkov2, weight: this.modelWeights.markov2.weight },
            { name: 'markov3', prob: probMarkov3, weight: this.modelWeights.markov3.weight },
            { name: 'logistic', prob: probLogistic, weight: this.modelWeights.logistic.weight },
            { name: 'naiveBayes', prob: probNB, weight: this.modelWeights.naiveBayes.weight }
        ];

        let weightedProb = 0, totalWeight = 0;
        for (let p of probs) {
            weightedProb += p.prob * p.weight;
            totalWeight += p.weight;
        }
        const finalProb = totalWeight > 0 ? weightedProb / totalWeight : 0.5;
        const tiLeTai = finalProb * 100;
        const ketQuaDuDoan = finalProb >= 0.5 ? 'tài' : 'xỉu';
        const soLieuSuDung = history.length;
        let doTinCay = 'trung bình';
        if (soLieuSuDung >= 50 && Math.abs(finalProb - 0.5) > 0.2) doTinCay = 'cao';
        else if (soLieuSuDung < 20) doTinCay = 'thấp';

        return {
            phien_du_doan: history.length > 0 ? history[history.length-1].sid + 1 : 1,
            ket_qua_du_doan: ketQuaDuDoan,
            ti_le_tai: tiLeTai.toFixed(2),
            ti_le_xiu: (100 - tiLeTai).toFixed(2),
            so_lieu_su_dung: soLieuSuDung,
            do_tin_cay: doTinCay,
            thoi_gian_du_doan: new Date().toISOString()
        };
    }
}

// ==================== LỚP THEO DÕI ĐỘ CHÍNH XÁC ====================
class AccuracyTracker {
    constructor() {
        this.stats = {
            tx: { total: 0, correct: 0, recent: [], brier: [] },
            md5: { total: 0, correct: 0, recent: [], brier: [] }
        };
    }

    update(table, actual, prediction) {
        const stat = this.stats[table];
        if (!stat) return;
        const actualBin = actual === 'tài' ? 1 : 0;
        const predBin = prediction.ket_qua_du_doan === 'tài' ? 1 : 0;
        const correct = (actualBin === predBin) ? 1 : 0;
        
        stat.total++;
        if (correct) stat.correct++;
        
        stat.recent.push(correct);
        if (stat.recent.length > 50) stat.recent.shift();

        const probTai = parseFloat(prediction.ti_le_tai) / 100;
        const brier = Math.pow(probTai - actualBin, 2);
        stat.brier.push(brier);
        if (stat.brier.length > 50) stat.brier.shift();
    }

    getAccuracy(table) {
        const stat = this.stats[table];
        if (!stat || stat.total === 0) return { overall: 0, recent: 0, brier: 0 };
        const overall = stat.correct / stat.total;
        const recent = stat.recent.length ? stat.recent.reduce((a,b)=>a+b,0) / stat.recent.length : 0;
        const brier = stat.brier.length ? stats.mean(stat.brier) : 0;
        return { overall, recent, brier };
    }

    getAllStats() {
        return {
            tx: this.getAccuracy('tx'),
            md5: this.getAccuracy('md5')
        };
    }
}

// ==================== MỞ RỘNG CLASS GAMEWEBSOCKETCLIENT ====================
class GameWebSocketClient {
    constructor(url) {
        this.url = url;
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 5000;
        this.isAuthenticated = false;
        this.sessionId = null;
        this.latestTxData = null;
        this.latestMd5Data = null;
        this.lastUpdateTime = { tx: null, md5: null };
        // Các thuộc tính mới
        this.historyManager = new HistoryManager(100);
        this.predictor = new Predictor();
        this.accuracyTracker = new AccuracyTracker();
        this.lastPrediction = { tx: null, md5: null };
    }

    connect() {
        console.log('🔗 Connecting to WebSocket server...');
        this.ws = new WebSocket(this.url, {
            headers: {
                'Host': 'api.jiusyss.me',
                'Origin': 'https://play.son789.site',
                'User-Agent': 'Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Mobile Safari/537.36',
                'Pragma': 'no-cache',
                'Cache-Control': 'no-cache',
                'Accept-Encoding': 'gzip, deflate, br',
                'Accept-Language': 'vi-VN,vi;q=0.9,fr-FR;q=0.8,fr;q=0.7,en-US;q=0.6,en;q=0.5',
                'Sec-WebSocket-Extensions': 'permessage-deflate; client_max_window_bits',
                'Sec-WebSocket-Version': '13'
            }
        });
        this.setupEventHandlers();
    }

    setupEventHandlers() {
        this.ws.on('open', () => {
            console.log('✅ Connected to WebSocket server');
            this.reconnectAttempts = 0;
            this.sendAuthentication();
        });

        this.ws.on('message', (data) => {
            this.handleMessage(data);
        });

        this.ws.on('error', (error) => {
            console.error('❌ WebSocket error:', error.message);
        });

        this.ws.on('close', (code, reason) => {
            console.log(`🔌 Connection closed. Code: ${code}, Reason: ${String(reason)}`);
            this.isAuthenticated = false;
            this.sessionId = null;
            this.handleReconnect();
        });

        this.ws.on('pong', () => {
            console.log('❤️ Heartbeat received from server');
        });
    }

    sendAuthentication() {
        console.log('🔐 Sending authentication...');
        const authMessage = [
            1,
            "MiniGame",
            "son789apia",
            "WangLin1@",
            {
                "signature": "3B807F3D9780682F163184B42F8A3B30B26814FF23F1B7784F99DC842AC076F758E4718F533AF9405F1129E3830A236DAAA0127F1EECA73BC6EB057B5174E4509D57408CCF2C7E316136F98CE46843E6920130C60465D474CABAF6F911E7068DE9B20198CFF684DE6270C9E42922A46E46F5D60EC2BAA9B75F9BE8605E824CA0",
                "info": {
                    "cs": "9e05a39a8958d83119db6ab9a1d88548",
                    "phone": "",
                    "ipAddress": "113.185.46.68",
                    "isMerchant": false,
                    "userId": "bf5dc66b-2e77-4b48-ab73-09f2ffbe3443",
                    "deviceId": "050105373613900053736078036024",
                    "isMktAccount": false,
                    "username": "son789apia",
                    "timestamp": 1766557267829
                },
                "pid": 4
            }
        ];
        this.sendRaw(authMessage);
    }

    sendPluginMessages() {
        console.log('🚀 Sending plugin initialization messages...');
        const pluginMessages = [
            [6,"MiniGame","taixiuPlugin",{"cmd":1005}],
            [6,"MiniGame","taixiuMd5Plugin",{"cmd":1105}],
            [6,"MiniGame","lobbyPlugin",{"cmd":10001}],
            [6,"MiniGame","channelPlugin",{"cmd":310}]
        ];
        pluginMessages.forEach((message, index) => {
            setTimeout(() => {
                console.log(`📤 Sending plugin ${index + 1}/${pluginMessages.length}: ${message[2]}`);
                this.sendRaw(message);
            }, index * 1000);
        });
        setInterval(() => { this.refreshGameData(); }, 30000);
    }

    refreshGameData() {
        if (this.isAuthenticated && this.ws && this.ws.readyState === WebSocket.OPEN) {
            console.log('🔄 Refreshing game data...');
            const refreshTx = [6, "MiniGame", "taixiuPlugin", { "cmd": 1005 }];
            const refreshMd5 = [6, "MiniGame", "taixiuMd5Plugin", { "cmd": 1105 }];
            this.sendRaw(refreshTx);
            setTimeout(() => { this.sendRaw(refreshMd5); }, 1000);
        }
    }

    sendRaw(data) {
        if (this.ws.readyState === WebSocket.OPEN) {
            const jsonString = JSON.stringify(data);
            this.ws.send(jsonString);
            console.log('📤 Sent raw:', jsonString);
            return true;
        } else {
            console.log('⚠️ Cannot send, WebSocket not open');
            return false;
        }
    }

    handleMessage(data) {
        try {
            const parsed = JSON.parse(data);
            
            // XỬ LÝ CMD 1005 - BÀN TÀI XỈU THƯỜNG
            if (parsed[0] === 5 && parsed[1] && parsed[1].cmd === 1005) {
                console.log('🎯 Nhận dữ liệu cmd 1005 (Bàn TX)');
                const gameData = parsed[1];
                if (gameData.htr && gameData.htr.length > 0) {
                    const added = this.historyManager.addSessionsFromGameData('tx', gameData);
                    if (added > 0) {
                        console.log(`📝 Đã thêm ${added} phiên mới cho bàn TX`);
                        this.latestTxData = gameData;
                        this.lastUpdateTime.tx = new Date();
                        if (this.historyManager.count('tx') >= 10) {
                            this.runPrediction('tx');
                        }
                    }
                }
            }
            
            // XỬ LÝ CMD 1105 - BÀN MD5
            else if (parsed[0] === 5 && parsed[1] && parsed[1].cmd === 1105) {
                console.log('🎯 Nhận dữ liệu cmd 1105 (Bàn MD5)');
                const gameData = parsed[1];
                if (gameData.htr && gameData.htr.length > 0) {
                    const added = this.historyManager.addSessionsFromGameData('md5', gameData);
                    if (added > 0) {
                        console.log(`📝 Đã thêm ${added} phiên mới cho bàn MD5`);
                        this.latestMd5Data = gameData;
                        this.lastUpdateTime.md5 = new Date();
                        if (this.historyManager.count('md5') >= 10) {
                            this.runPrediction('md5');
                        }
                    }
                }
            }
            
            // Xử lý response authentication (type 5, có cmd 100)
            else if (parsed[0] === 5 && parsed[1] && parsed[1].cmd === 100) {
                console.log('🔑 Authentication successful!');
                const userData = parsed[1];
                console.log(`✅ User: ${userData.u}`);
                this.isAuthenticated = true;
                setTimeout(() => {
                    console.log('🔄 Starting to send plugin messages...');
                    this.sendPluginMessages();
                }, 2000);
            }
            
            // Xử lý response type 1 - Session initialization
            else if (parsed[0] === 1 && parsed.length >= 5 && parsed[4] === "MiniGame") {
                console.log('✅ Session initialized');
                this.sessionId = parsed[3];
                console.log(`📋 Session ID: ${this.sessionId}`);
            }
            
            // Xử lý response type 7 - Plugin response
            else if (parsed[0] === 7) {
                const pluginName = parsed[2];
                console.log(`🔄 Plugin ${pluginName} response received`);
            }
            
            // Xử lý heartbeat/ping response
            else if (parsed[0] === 0) {
                console.log('❤️ Heartbeat received');
            }
            
        } catch (e) {
            console.log('📥 Raw message:', data.toString());
            console.error('❌ Parse error:', e.message);
        }
    }

    runPrediction(table) {
        setTimeout(() => {
            console.log(`🔮 Đang dự đoán cho bàn ${table}...`);
            const pred = this.predictor.predict(table, this.historyManager);
            if (!pred.error) {
                this.lastPrediction[table] = pred;
                console.log(`✅ Dự đoán ${table}: ${pred.ket_qua_du_doan} (tài: ${pred.ti_le_tai}%, xỉu: ${pred.ti_le_xiu}%)`);
            } else {
                console.log(`⚠️ Không thể dự đoán ${table}: ${pred.error}`);
            }
        }, 0);
    }

    getLatestTxSession() {
        if (!this.latestTxData || !this.latestTxData.htr || this.latestTxData.htr.length === 0) {
            return { error: "Không có dữ liệu bàn TX" };
        }
        try {
            const latestSession = this.latestTxData.htr.reduce((prev, current) => (current.sid > prev.sid) ? current : prev);
            const tong = latestSession.d1 + latestSession.d2 + latestSession.d3;
            const ket_qua = (tong >= 11) ? "tài" : "xỉu";
            return {
                phien: latestSession.sid,
                xuc_xac_1: latestSession.d1,
                xuc_xac_2: latestSession.d2,
                xuc_xac_3: latestSession.d3,
                tong: tong,
                ket_qua: ket_qua,
                timestamp: new Date().toISOString(),
                ban: "tai_xiu",
                last_updated: this.lastUpdateTime.tx ? this.lastUpdateTime.tx.toISOString() : null
            };
        } catch (error) {
            return { error: "Lỗi xử lý dữ liệu TX", message: error.message };
        }
    }

    getLatestMd5Session() {
        if (!this.latestMd5Data || !this.latestMd5Data.htr || this.latestMd5Data.htr.length === 0) {
            return { error: "Không có dữ liệu bàn MD5" };
        }
        try {
            const latestSession = this.latestMd5Data.htr.reduce((prev, current) => (current.sid > prev.sid) ? current : prev);
            const tong = latestSession.d1 + latestSession.d2 + latestSession.d3;
            const ket_qua = (tong >= 11) ? "tài" : "xỉu";
            return {
                phien: latestSession.sid,
                xuc_xac_1: latestSession.d1,
                xuc_xac_2: latestSession.d2,
                xuc_xac_3: latestSession.d3,
                tong: tong,
                ket_qua: ket_qua,
                timestamp: new Date().toISOString(),
                ban: "md5",
                last_updated: this.lastUpdateTime.md5 ? this.lastUpdateTime.md5.toISOString() : null
            };
        } catch (error) {
            return { error: "Lỗi xử lý dữ liệu MD5", message: error.message };
        }
    }

    handleReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            const delay = this.reconnectDelay * this.reconnectAttempts;
            console.log(`🔄 Attempting to reconnect in ${delay}ms (Attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
            setTimeout(() => {
                console.log('🔄 Reconnecting...');
                this.connect();
            }, delay);
        } else {
            console.log('❌ Max reconnection attempts reached');
        }
    }

    startHeartbeat() {
        setInterval(() => {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                const heartbeatMsg = [0, this.sessionId || ""];
                this.sendRaw(heartbeatMsg);
                console.log('❤️ Sending heartbeat...');
            }
        }, 25000);
    }

    close() {
        if (this.ws) {
            this.ws.close();
        }
    }
}

// ==================== KHỞI TẠO EXPRESS SERVER ====================
const app = express();
const PORT = 3000;
app.use(cors());
app.use(express.json());

// Tạo WebSocket client
const client = new GameWebSocketClient(
    'wss://api.jiusyss.me/websocket?d=YUd0aGIyNWliMmM9fDEyOTh8MTc2NjU1NzI2NzI2M3wzMmNlMmE1NGQzNmFhY2FhMWZmNjZmMzE5MzQ1ZmUyNXw5MjJjMjBhMTE4NTBiNzRiNmNjYzQxMTE3Nzk0NDQ5Zg=='
);
client.connect();

// ==================== ROUTES HIỆN TẠI ====================
app.get('/api/tx', (req, res) => {
    const data = client.getLatestTxSession();
    if (data.error) return res.status(404).json(data);
    res.json(data);
});

app.get('/api/md5', (req, res) => {
    const data = client.getLatestMd5Session();
    if (data.error) return res.status(404).json(data);
    res.json(data);
});

app.get('/api/all', (req, res) => {
    const txSession = client.getLatestTxSession();
    const md5Session = client.getLatestMd5Session();
    res.json({
        tai_xiu: txSession.error ? { error: txSession.error } : txSession,
        md5: md5Session.error ? { error: md5Session.error } : md5Session,
        timestamp: new Date().toISOString()
    });
});

app.get('/api/status', (req, res) => {
    const hasTxData = client.latestTxData && client.latestTxData.htr && client.latestTxData.htr.length > 0;
    const hasMd5Data = client.latestMd5Data && client.latestMd5Data.htr && client.latestMd5Data.htr.length > 0;
    res.json({
        status: "running",
        websocket_connected: client.ws ? client.ws.readyState === WebSocket.OPEN : false,
        authenticated: client.isAuthenticated,
        has_tx_data: hasTxData,
        has_md5_data: hasMd5Data,
        tx_last_updated: client.lastUpdateTime.tx ? client.lastUpdateTime.tx.toISOString() : null,
        md5_last_updated: client.lastUpdateTime.md5 ? client.lastUpdateTime.md5.toISOString() : null,
        timestamp: new Date().toISOString()
    });
});

app.get('/api/refresh', (req, res) => {
    if (client.isAuthenticated && client.ws && client.ws.readyState === WebSocket.OPEN) {
        client.refreshGameData();
        res.json({ message: "Đã gửi yêu cầu refresh dữ liệu", timestamp: new Date().toISOString() });
    } else {
        res.status(400).json({ error: "Không thể refresh", message: "WebSocket chưa kết nối hoặc chưa xác thực" });
    }
});

// ==================== ROUTES MỚI ====================
app.get('/api/predict/tx', (req, res) => {
    if (!client.lastPrediction.tx) {
        return res.status(404).json({ error: "Chưa có dự đoán cho bàn TX" });
    }
    res.json(client.lastPrediction.tx);
});

app.get('/api/predict/md5', (req, res) => {
    if (!client.lastPrediction.md5) {
        return res.status(404).json({ error: "Chưa có dự đoán cho bàn MD5" });
    }
    res.json(client.lastPrediction.md5);
});

app.get('/api/predict/all', (req, res) => {
    res.json({
        tx: client.lastPrediction.tx || { error: "Chưa có dự đoán TX" },
        md5: client.lastPrediction.md5 || { error: "Chưa có dự đoán MD5" },
        timestamp: new Date().toISOString()
    });
});

app.get('/api/accuracy', (req, res) => {
    res.json(client.accuracyTracker.getAllStats());
});

app.get('/api/history/:ban', (req, res) => {
    const ban = req.params.ban;
    if (ban !== 'tx' && ban !== 'md5') {
        return res.status(400).json({ error: "Bàn không hợp lệ, chỉ tx hoặc md5" });
    }
    const history = client.historyManager.getHistory(ban, 100);
    res.json({ ban, history, count: history.length });
});

app.get('/api/stats/:ban', (req, res) => {
    const ban = req.params.ban;
    if (ban !== 'tx' && ban !== 'md5') {
        return res.status(400).json({ error: "Bàn không hợp lệ" });
    }
    const history = client.historyManager.getHistory(ban, 100);
    if (history.length === 0) {
        return res.json({ ban, message: "Chưa có dữ liệu" });
    }
    const results = history.map(s => s.ket_qua);
    const taiCount = results.filter(r => r === 'tài').length;
    const xiuCount = results.length - taiCount;
    const tongs = history.map(s => s.tong);
    const mean = stats.mean(tongs);
    const std = stats.standardDeviation(tongs);
    res.json({
        ban,
        total_sessions: history.length,
        tai_ratio: (taiCount / results.length).toFixed(3),
        xiu_ratio: (xiuCount / results.length).toFixed(3),
        tong_mean: mean.toFixed(2),
        tong_std: std.toFixed(2),
        last_updated: client.lastUpdateTime[ban]?.toISOString() || null
    });
});

app.get('/', (req, res) => {
    res.send(`
        <html>
            <head><title>API Status</title></head>
            <body>
                <h1>API is running</h1>
                <p>Endpoints:</p>
                <ul>
                    <li>/api/tx - Dữ liệu mới nhất bàn TX</li>
                    <li>/api/md5 - Dữ liệu mới nhất bàn MD5</li>
                    <li>/api/all - Cả hai bàn</li>
                    <li>/api/status - Trạng thái kết nối</li>
                    <li>/api/refresh - Refresh dữ liệu thủ công</li>
                    <li>/api/predict/tx - Dự đoán bàn TX</li>
                    <li>/api/predict/md5 - Dự đoán bàn MD5</li>
                    <li>/api/predict/all - Dự đoán cả hai</li>
                    <li>/api/accuracy - Độ chính xác</li>
                    <li>/api/history/:ban - Lịch sử phiên (tx/md5)</li>
                    <li>/api/stats/:ban - Thống kê cơ bản</li>
                </ul>
            </body>
        </html>
    `);
});

app.listen(PORT, '0.0.0.0', () => {
    console.log(`🚀 Server đang chạy tại: http://localhost:${PORT}`);
});

setTimeout(() => {
    client.startHeartbeat();
}, 10000);

process.on('SIGINT', () => {
    console.log('\n👋 Closing WebSocket connection and server...');
    client.close();
    process.exit();
});

module.exports = { GameWebSocketClient, app };
