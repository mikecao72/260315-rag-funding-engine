(function () {
    "use strict";

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
            if (btn.dataset.tab === "db-inspector") loadDbSchedules();
        });
    });

    // ── Top-N slider ──
    var topNSlider = document.getElementById("top-n");
    var topNValue = document.getElementById("top-n-value");
    topNSlider.addEventListener("input", function () {
        topNValue.textContent = topNSlider.value;
    });

    // ── Min Confidence slider ──
    var confidenceSlider = document.getElementById("min-confidence");
    var confidenceValue = document.getElementById("confidence-value");
    confidenceSlider.addEventListener("input", function () {
        confidenceValue.textContent = parseFloat(confidenceSlider.value).toFixed(1);
    });

    // ── Load schedules ──
    function loadSchedules() {
        fetch("/schedules")
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
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                consult_text: consultText,
                schedule_id: scheduleId,
                top_n: parseInt(topNSlider.value, 10),
                gst_mode: document.getElementById("gst-mode").value,
                min_confidence: parseFloat(confidenceSlider.value)
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
                '<td><input type="number" class="qty-input" data-code="' + esc(row.code) + '" min="1" value="' + qty + '"></td>' +
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
                quantities[code] = Math.max(1, parseInt(inp.value, 10) || 1);
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

        fetch("/ingest", { method: "POST", body: fd })
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

    // ── DB Inspector ──
    var dbScheduleSelect = document.getElementById("db-schedule-select");
    var dbTableList = document.getElementById("db-table-list");
    var dbState = { scheduleId: "", tables: [], activeTable: "", offset: 0, pageSize: 50, total: 0 };

    function loadDbSchedules() {
        fetch("/schedules")
            .then(function (r) { return r.json(); })
            .then(function (data) {
                var schedules = data.schedules || [];
                dbScheduleSelect.innerHTML = "";
                if (schedules.length === 0) {
                    dbScheduleSelect.innerHTML = '<option value="">No schedules</option>';
                    return;
                }
                schedules.forEach(function (s) {
                    var opt = document.createElement("option");
                    opt.value = s.schedule_id;
                    opt.textContent = s.schedule_id;
                    dbScheduleSelect.appendChild(opt);
                });
                if (schedules.length > 0) {
                    dbState.scheduleId = schedules[0].schedule_id;
                    loadDbTables();
                }
            })
            .catch(function () {});
    }

    dbScheduleSelect.addEventListener("change", function () {
        dbState.scheduleId = dbScheduleSelect.value;
        dbState.activeTable = "";
        resetDbMain();
        loadDbTables();
    });

    function loadDbTables() {
        if (!dbState.scheduleId) return;
        dbTableList.innerHTML = '<div style="color:#71717a;font-size:0.8rem;padding:0.5rem 0">Loading...</div>';
        fetch("/db/" + encodeURIComponent(dbState.scheduleId) + "/tables")
            .then(function (r) { return r.json(); })
            .then(function (data) {
                if (data.error) {
                    dbTableList.innerHTML = '<div class="error-msg">' + esc(data.error) + '</div>';
                    return;
                }
                dbState.tables = data.tables || [];
                renderDbTableList();
            })
            .catch(function (err) {
                dbTableList.innerHTML = '<div class="error-msg">' + esc(err.message) + '</div>';
            });
    }

    function renderDbTableList() {
        if (dbState.tables.length === 0) {
            dbTableList.innerHTML = '<div class="empty-state" style="padding:1rem 0">No tables found.</div>';
            return;
        }
        var html = "";
        dbState.tables.forEach(function (t) {
            var cls = t.name === dbState.activeTable ? "db-table-btn active" : "db-table-btn";
            html += '<button type="button" class="' + cls + '" data-table="' + esc(t.name) + '">' +
                '<span class="db-table-name">' + esc(t.name) + '</span>' +
                '<span class="db-table-count">' + t.row_count + ' rows</span>' +
                '</button>';
        });
        dbTableList.innerHTML = html;

        dbTableList.querySelectorAll(".db-table-btn").forEach(function (btn) {
            btn.addEventListener("click", function () {
                dbState.activeTable = btn.dataset.table;
                dbState.offset = 0;
                renderDbTableList();
                loadDbTableData();
            });
        });
    }

    function resetDbMain() {
        document.getElementById("db-schema-section").style.display = "none";
        document.getElementById("db-rows-section").style.display = "none";
        document.getElementById("db-empty-state").style.display = "";
        document.getElementById("db-table-label").textContent = "Select a table";
        document.getElementById("db-table-title").textContent = "\u2014";
        document.getElementById("db-row-count").textContent = "";
    }

    function loadDbTableData() {
        var tableName = dbState.activeTable;
        if (!tableName || !dbState.scheduleId) return;

        // Find schema from cached tables
        var tableInfo = null;
        dbState.tables.forEach(function (t) { if (t.name === tableName) tableInfo = t; });
        if (tableInfo) renderDbSchema(tableInfo);

        // Fetch rows
        var url = "/db/" + encodeURIComponent(dbState.scheduleId) +
            "/tables/" + encodeURIComponent(tableName) +
            "?limit=" + dbState.pageSize + "&offset=" + dbState.offset;

        fetch(url)
            .then(function (r) { return r.json(); })
            .then(function (data) {
                if (data.error) return;
                dbState.total = data.total;
                renderDbRows(data);
            })
            .catch(function () {});
    }

    function renderDbSchema(tableInfo) {
        document.getElementById("db-empty-state").style.display = "none";
        document.getElementById("db-schema-section").style.display = "";
        document.getElementById("db-table-label").textContent = "Table";
        document.getElementById("db-table-title").textContent = tableInfo.name;
        document.getElementById("db-row-count").textContent = tableInfo.row_count + " total rows";

        var body = document.getElementById("db-schema-body");
        var html = "";
        tableInfo.columns.forEach(function (col) {
            html += '<tr>' +
                '<td><strong>' + esc(col.name) + '</strong></td>' +
                '<td><span class="db-type-badge">' + esc(col.type || "—") + '</span></td>' +
                '<td>' + (col.pk ? '<span class="db-pk-badge">PK</span>' : '') + '</td>' +
                '<td>' + (col.notnull ? 'Yes' : '') + '</td>' +
                '</tr>';
        });
        body.innerHTML = html;
    }

    function renderDbRows(data) {
        document.getElementById("db-rows-section").style.display = "";
        var columns = data.columns || [];
        var rows = data.rows || [];

        // Header
        var headHtml = '<tr>';
        columns.forEach(function (col) {
            headHtml += '<th>' + esc(col) + '</th>';
        });
        headHtml += '</tr>';
        document.getElementById("db-rows-head").innerHTML = headHtml;

        // Body
        var bodyHtml = "";
        if (rows.length === 0) {
            bodyHtml = '<tr><td colspan="' + columns.length + '" class="empty-cell">No rows.</td></tr>';
        } else {
            rows.forEach(function (row) {
                bodyHtml += '<tr>';
                columns.forEach(function (col) {
                    var val = row[col];
                    var display = val == null ? '<span style="color:#52525b">NULL</span>' : esc(String(val));
                    bodyHtml += '<td title="' + esc(String(val != null ? val : "")) + '">' + display + '</td>';
                });
                bodyHtml += '</tr>';
            });
        }
        document.getElementById("db-rows-body").innerHTML = bodyHtml;

        // Pagination
        var total = data.total;
        var start = data.offset + 1;
        var end = Math.min(data.offset + data.rows.length, total);
        document.getElementById("db-page-info").textContent = start + "–" + end + " of " + total;

        var prevBtn = document.getElementById("db-prev-btn");
        var nextBtn = document.getElementById("db-next-btn");
        prevBtn.disabled = data.offset === 0;
        nextBtn.disabled = (data.offset + dbState.pageSize) >= total;
    }

    document.getElementById("db-prev-btn").addEventListener("click", function () {
        dbState.offset = Math.max(0, dbState.offset - dbState.pageSize);
        loadDbTableData();
    });

    document.getElementById("db-next-btn").addEventListener("click", function () {
        dbState.offset += dbState.pageSize;
        loadDbTableData();
    });

    // ── Helpers ──
    function esc(str) {
        if (str == null) return "";
        var d = document.createElement("div");
        d.appendChild(document.createTextNode(String(str)));
        return d.innerHTML;
    }

    // ── Init ──
    loadSchedules();
})();
