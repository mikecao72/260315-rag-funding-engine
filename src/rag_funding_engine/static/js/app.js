(function () {
    "use strict";

    // ── Auth ──
    var authToken = sessionStorage.getItem("authToken") || "";

    function authHeaders() {
        return { "Authorization": "Basic " + authToken };
    }

    function showLogin() {
        document.getElementById("login-overlay").style.display = "flex";
        document.getElementById("app-content").style.display = "none";
    }

    function hideLogin() {
        document.getElementById("login-overlay").style.display = "none";
        document.getElementById("app-content").style.display = "block";
    }

    document.getElementById("login-form").addEventListener("submit", function (e) {
        e.preventDefault();
        var user = document.getElementById("login-user").value;
        var pass = document.getElementById("login-pass").value;
        authToken = btoa(user + ":" + pass);

        // Test credentials against /schedules
        fetch("/schedules", { headers: authHeaders() })
            .then(function (r) {
                if (r.status === 401) {
                    document.getElementById("login-error").textContent = "Invalid username or password";
                    document.getElementById("login-error").hidden = false;
                    return;
                }
                sessionStorage.setItem("authToken", authToken);
                document.getElementById("login-error").hidden = true;
                hideLogin();
                return r.json();
            })
            .then(function (data) {
                if (data) {
                    populateDropdown(data.schedules || []);
                    renderSchedulesList(data.schedules || []);
                }
            })
            .catch(function () {
                document.getElementById("login-error").textContent = "Connection failed";
                document.getElementById("login-error").hidden = false;
            });
    });

    document.getElementById("logout-btn").addEventListener("click", function () {
        authToken = "";
        sessionStorage.removeItem("authToken");
        showLogin();
    });

    // ── State ──
    var results = [];
    var quantities = {};
    var expandedCode = null;

    // ── Tab navigation ──
    var tabs = document.querySelectorAll(".tab");
    var sections = document.querySelectorAll(".tab-content");

    tabs.forEach(function (btn) {
        btn.addEventListener("click", function () {
            tabs.forEach(function (t) { t.classList.remove("active"); });
            sections.forEach(function (s) { s.classList.remove("active"); });
            btn.classList.add("active");
            document.getElementById(btn.dataset.tab).classList.add("active");
            if (btn.dataset.tab === "schedules") loadSchedules();
        });
    });

    // ── Top-N slider ──
    var topNSlider = document.getElementById("top-n");
    var topNValue = document.getElementById("top-n-value");
    topNSlider.addEventListener("input", function () {
        topNValue.textContent = topNSlider.value;
    });

    // ── Load schedules ──
    function loadSchedules() {
        fetch("/schedules", { headers: authHeaders() })
            .then(function (r) { return r.json(); })
            .then(function (data) {
                populateDropdown(data.schedules || []);
                renderSchedulesList(data.schedules || []);
            })
            .catch(function () {});
    }

    function populateDropdown(schedules) {
        var sel = document.getElementById("schedule-select");
        sel.innerHTML = "";
        if (schedules.length === 0) {
            sel.innerHTML = '<option value="">No schedules ingested</option>';
            return;
        }
        schedules.forEach(function (s) {
            var opt = document.createElement("option");
            opt.value = s.schedule_id;
            opt.textContent = s.schedule_id + (s.description ? " — " + s.description : "");
            sel.appendChild(opt);
        });
    }

    function renderSchedulesList(schedules) {
        var el = document.getElementById("schedules-list");
        if (schedules.length === 0) {
            el.innerHTML = '<div class="empty-state">No schedules ingested yet. Use the Ingest tab to upload a PDF.</div>';
            return;
        }
        el.innerHTML = schedules.map(function (s) {
            return '<div class="schedule-card">' +
                '<h3>' + esc(s.schedule_id) + '</h3>' +
                '<div class="schedule-meta">' +
                (s.description ? esc(s.description) + '<br>' : '') +
                (s.schedule_type ? 'Type: ' + esc(s.schedule_type) + ' · ' : '') +
                'Codes: ' + (s.codes_parsed || 0) + ' · Chunks: ' + (s.chunks_indexed || 0) +
                '</div></div>';
        }).join("");
    }

    // ── Recommend ──
    var recommendBtn = document.getElementById("recommend-btn");
    var errorEl = document.getElementById("recommend-error");

    recommendBtn.addEventListener("click", function () {
        var consultText = document.getElementById("consult-text").value.trim();
        if (!consultText) return;

        var scheduleId = document.getElementById("schedule-select").value;
        if (!scheduleId) {
            showError("No schedule selected. Ingest a schedule first.");
            return;
        }

        recommendBtn.disabled = true;
        recommendBtn.textContent = "Running...";
        errorEl.hidden = true;
        showLoading();

        fetch("/recommend", {
            method: "POST",
            headers: { "Content-Type": "application/json", "Authorization": "Basic " + authToken },
            body: JSON.stringify({
                consult_text: consultText,
                schedule_id: scheduleId,
                top_n: parseInt(topNSlider.value, 10),
                gst_mode: document.getElementById("gst-mode").value
            })
        })
        .then(function (r) { return r.json(); })
        .then(function (data) {
            recommendBtn.disabled = false;
            recommendBtn.textContent = "Suggest Billing Codes";

            if (data.error) {
                showError(data.error);
                results = [];
                renderTable();
                return;
            }

            results = data.recommendations || data.codes || [];
            expandedCode = null;
            quantities = {};
            results.forEach(function (r) { quantities[r.code] = 1; });
            renderTable();
        })
        .catch(function (err) {
            recommendBtn.disabled = false;
            recommendBtn.textContent = "Suggest Billing Codes";
            showError(err.message || "Request failed");
            results = [];
            renderTable();
        });
    });

    function showError(msg) {
        errorEl.textContent = msg;
        errorEl.hidden = false;
    }

    function showLoading() {
        var body = document.getElementById("results-body");
        body.innerHTML = '<tr class="loading-row"><td colspan="4">Analysing consultation<span class="spinner"></span></td></tr>';
        document.getElementById("result-count").textContent = "...";
    }

    function renderTable() {
        var body = document.getElementById("results-body");
        document.getElementById("result-count").textContent = results.length + " suggested code(s)";

        if (results.length === 0) {
            body.innerHTML = '<tr><td colspan="4" class="empty-cell">No suggestions yet.</td></tr>';
            updateGrandTotal();
            return;
        }

        var html = "";
        results.forEach(function (row) {
            var qty = quantities[row.code] || 1;
            var unit = row.fee != null ? row.fee : (row.fee_excl_gst || 0);
            var total = unit * qty;
            var expanded = expandedCode === row.code;

            html += '<tr>' +
                '<td>' +
                    '<button type="button" class="code-btn" data-code="' + esc(row.code) + '">' +
                        '<div class="code-name">' + esc(row.code) + '</div>' +
                        '<div class="code-desc">' + esc(row.description || row.code_description || "") + '</div>' +
                        '<div class="code-toggle">' + (expanded ? "Hide reasoning &#9650;" : "Show reasoning &#9660;") + '</div>' +
                    '</button>' +
                '</td>' +
                '<td>$' + unit.toFixed(2) + '</td>' +
                '<td><input type="number" class="qty-input" data-code="' + esc(row.code) + '" min="0.1" step="0.1" value="' + qty + '"></td>' +
                '<td>$' + total.toFixed(2) + '</td>' +
            '</tr>';

            if (expanded) {
                html += '<tr class="reason-row"><td colspan="4">' +
                    esc(row.reason || "No reasoning available yet.") +
                '</td></tr>';
            }
        });

        body.innerHTML = html;
        updateGrandTotal();

        // Bind toggle buttons
        body.querySelectorAll(".code-btn").forEach(function (btn) {
            btn.addEventListener("click", function () {
                var code = btn.dataset.code;
                expandedCode = (expandedCode === code) ? null : code;
                renderTable();
            });
        });

        // Bind qty inputs
        body.querySelectorAll(".qty-input").forEach(function (inp) {
            inp.addEventListener("change", function () {
                var code = inp.dataset.code;
                quantities[code] = Math.max(0.1, parseFloat(inp.value) || 1);
                renderTable();
            });
        });
    }

    function updateGrandTotal() {
        var total = 0;
        results.forEach(function (row) {
            var qty = quantities[row.code] || 1;
            var unit = row.fee != null ? row.fee : (row.fee_excl_gst || 0);
            total += unit * qty;
        });
        document.querySelector(".grand-total-amount").textContent = "$" + total.toFixed(2);
    }

    // ── Ingest ──
    var ingestForm = document.getElementById("ingest-form");
    ingestForm.addEventListener("submit", function (e) {
        e.preventDefault();

        var scheduleId = document.getElementById("ingest-schedule-id").value.trim();
        var fileInput = document.getElementById("ingest-pdf");
        var model = document.getElementById("ingest-model").value.trim();
        if (!scheduleId || !fileInput.files.length) return;

        var fd = new FormData();
        fd.append("schedule_id", scheduleId);
        fd.append("pdf", fileInput.files[0]);
        fd.append("llm_model", model);

        var btn = document.getElementById("ingest-btn");
        btn.disabled = true;
        btn.textContent = "Ingesting...";
        document.getElementById("ingest-results").innerHTML =
            '<div style="margin-top:0.75rem;color:#71717a">Ingesting schedule (this may take a minute)<span class="spinner"></span></div>';

        fetch("/ingest", { method: "POST", body: fd, headers: authHeaders() })
            .then(function (r) { return r.json(); })
            .then(function (data) {
                btn.disabled = false;
                btn.textContent = "Ingest Schedule";
                if (data.status === "ok") {
                    document.getElementById("ingest-results").innerHTML =
                        '<div class="success-msg">Ingested <strong>' + esc(data.schedule_id) +
                        '</strong>: ' + data.code_count + ' codes, ' + data.indexed_chunks + ' chunks.</div>';
                    loadSchedules();
                } else {
                    document.getElementById("ingest-results").innerHTML =
                        '<div class="error-msg">' + esc(JSON.stringify(data)) + '</div>';
                }
            })
            .catch(function (err) {
                btn.disabled = false;
                btn.textContent = "Ingest Schedule";
                document.getElementById("ingest-results").innerHTML =
                    '<div class="error-msg">Upload failed: ' + esc(err.message) + '</div>';
            });
    });

    // ── Helpers ──
    function esc(str) {
        if (str == null) return "";
        var d = document.createElement("div");
        d.appendChild(document.createTextNode(String(str)));
        return d.innerHTML;
    }

    // ── Init ──
    if (authToken) {
        // Validate stored credentials
        fetch("/schedules", { headers: authHeaders() })
            .then(function (r) {
                if (r.status === 401) {
                    sessionStorage.removeItem("authToken");
                    authToken = "";
                    showLogin();
                    return;
                }
                hideLogin();
                return r.json();
            })
            .then(function (data) {
                if (data) {
                    populateDropdown(data.schedules || []);
                    renderSchedulesList(data.schedules || []);
                }
            })
            .catch(function () { showLogin(); });
    } else {
        showLogin();
    }
})();
