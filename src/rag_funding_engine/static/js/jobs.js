(function () {
    "use strict";

    // ── State ──
    var allJobs = [];
    var currentJobId = null;
    var currentJobData = null;
    var expandedCode = null;

    // ── Helpers ──
    function esc(str) {
        if (str == null) return "";
        var d = document.createElement("div");
        d.appendChild(document.createTextNode(String(str)));
        return d.innerHTML;
    }

    function formatTimestamp(ts) {
        if (!ts) return "—";
        var d = new Date(ts);
        var date = d.toLocaleDateString("en-NZ", { day: "2-digit", month: "short", year: "2-digit" });
        var time = d.toLocaleTimeString("en-NZ", { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });
        return date + " " + time;
    }

    function formatDuration(s) {
        if (s == null) return "—";
        return s.toFixed(1) + "s";
    }

    // ── Load jobs list ──
    function loadJobs() {
        fetch("/jobs")
            .then(function (r) { return r.json(); })
            .then(function (data) {
                allJobs = data.jobs || [];
                renderJobsList();
            })
            .catch(function () {
                document.getElementById("jobs-body").innerHTML =
                    '<tr><td colspan="7" class="empty-cell">Failed to load jobs.</td></tr>';
            });
    }

    function renderJobsList() {
        var filter = (document.getElementById("jobs-search").value || "").toLowerCase();
        var sort = document.getElementById("jobs-sort").value;

        var filtered = allJobs.filter(function (j) {
            if (!filter) return true;
            var haystack = [j.job_id, j.schedule_id, j.description, j.timestamp].join(" ").toLowerCase();
            return haystack.indexOf(filter) !== -1;
        });

        // Sort
        filtered.sort(function (a, b) {
            if (sort === "newest") return (b.timestamp || "").localeCompare(a.timestamp || "");
            if (sort === "oldest") return (a.timestamp || "").localeCompare(b.timestamp || "");
            if (sort === "schedule") return (a.schedule_id || "").localeCompare(b.schedule_id || "");
            if (sort === "codes-desc") return (b.num_recommendations || 0) - (a.num_recommendations || 0);
            if (sort === "total-desc") return (b.total_excl_gst || 0) - (a.total_excl_gst || 0);
            return 0;
        });

        var body = document.getElementById("jobs-body");
        var empty = document.getElementById("jobs-empty");

        if (filtered.length === 0) {
            body.innerHTML = "";
            empty.style.display = "";
            return;
        }

        empty.style.display = "none";
        var html = "";
        filtered.forEach(function (j) {
            var rowClass = "jobs-row" + (j.has_error ? " job-error-row" : "");
            var total = j.total_excl_gst != null ? "$" + j.total_excl_gst.toFixed(2) : "—";
            html += '<tr class="' + rowClass + '" data-job-id="' + esc(j.job_id) + '">' +
                '<td><span class="job-id-text">' + esc(j.job_id) + '</span></td>' +
                '<td>' + esc(formatTimestamp(j.timestamp)) + '</td>' +
                '<td>' + esc(j.schedule_id) + '</td>' +
                '<td><span class="job-desc" title="' + esc(j.description) + '">' + esc(j.description) + '</span></td>' +
                '<td class="col-codes">' + (j.num_recommendations || 0) + '</td>' +
                '<td class="col-total">' + total + '</td>' +
                '<td class="col-dur">' + formatDuration(j.duration_seconds) + '</td>' +
                '</tr>';
        });
        body.innerHTML = html;

        // Bind row clicks
        body.querySelectorAll(".jobs-row").forEach(function (row) {
            row.addEventListener("click", function () {
                openJob(row.dataset.jobId);
            });
        });
    }

    // ── Filters ──
    document.getElementById("jobs-search").addEventListener("input", renderJobsList);
    document.getElementById("jobs-sort").addEventListener("change", renderJobsList);

    // ── Open a specific job ──
    function openJob(jobId) {
        currentJobId = jobId;
        expandedCode = null;

        document.getElementById("jobs-list-view").style.display = "none";
        document.getElementById("job-detail-view").style.display = "";

        document.getElementById("detail-job-id").textContent = jobId;
        document.getElementById("detail-schedule").textContent = "Loading...";
        document.getElementById("detail-time").textContent = "";
        document.getElementById("detail-consult-text").value = "Loading...";
        document.getElementById("detail-results-body").innerHTML =
            '<tr><td colspan="4" class="empty-cell">Loading...</td></tr>';
        document.getElementById("audit-log-content").textContent = "Loading...";

        // Activate first tab
        activateJobTab("input-output");

        // Fetch job data and log in parallel
        fetch("/jobs/" + encodeURIComponent(jobId))
            .then(function (r) { return r.json(); })
            .then(function (data) {
                if (data.error) {
                    document.getElementById("detail-consult-text").value = "Error: " + data.error;
                    return;
                }
                currentJobData = data;
                renderJobDetail(data);
            })
            .catch(function (err) {
                document.getElementById("detail-consult-text").value = "Failed to load: " + err.message;
            });

        fetch("/jobs/" + encodeURIComponent(jobId) + "/log")
            .then(function (r) { return r.json(); })
            .then(function (data) {
                document.getElementById("audit-log-content").textContent = data.log || data.error || "No log available.";
            })
            .catch(function () {
                document.getElementById("audit-log-content").textContent = "Failed to load audit log.";
            });
    }

    function renderJobDetail(data) {
        var inp = data.input || {};
        var out = data.output || {};

        // Meta
        document.getElementById("detail-schedule").textContent = inp.schedule_id || "—";
        document.getElementById("detail-time").textContent = formatTimestamp(data.timestamp);

        // Input
        document.getElementById("detail-consult-text").value = inp.consult_text || "";
        document.getElementById("detail-schedule-id").value = inp.schedule_id || "";
        document.getElementById("detail-gst-mode").value = (inp.gst_mode || "excl").toUpperCase();
        document.getElementById("detail-top-n").value = inp.top_n || "";
        document.getElementById("detail-min-conf").value = inp.min_confidence || "";

        // Output
        var recs = out.recommendations || [];
        document.getElementById("detail-result-count").textContent = recs.length + " suggested code(s)";
        renderDetailTable(recs);
    }

    function renderDetailTable(recs) {
        var body = document.getElementById("detail-results-body");

        if (recs.length === 0) {
            body.innerHTML = '<tr><td colspan="4" class="empty-cell">No recommendations.</td></tr>';
            updateDetailTotal(recs);
            return;
        }

        var html = "";
        recs.forEach(function (row) {
            var unit = row.fee != null ? row.fee : (row.fee_excl_gst || 0);
            var qty = row.quantity || 1;
            var lineTotal = row.line_total_excl_gst != null ? row.line_total_excl_gst : (unit * qty);
            var expanded = expandedCode === row.code;

            html += '<tr>' +
                '<td>' +
                    '<button type="button" class="code-btn" data-code="' + esc(row.code) + '">' +
                        '<div class="code-name">' + esc(row.code) + '</div>' +
                        '<div class="code-desc">' + esc(row.description || "") + '</div>' +
                        '<div class="code-toggle">' + (expanded ? "Hide reasoning &#9650;" : "Show reasoning &#9660;") + '</div>' +
                    '</button>' +
                '</td>' +
                '<td>$' + unit.toFixed(2) + '</td>' +
                '<td>' + qty + '</td>' +
                '<td>$' + lineTotal.toFixed(2) + '</td>' +
                '</tr>';

            if (expanded) {
                html += '<tr class="reason-row"><td colspan="4">' +
                    esc(row.reason || "No reasoning available.") +
                    '</td></tr>';
            }
        });

        body.innerHTML = html;
        updateDetailTotal(recs);

        // Bind toggle
        body.querySelectorAll(".code-btn").forEach(function (btn) {
            btn.addEventListener("click", function () {
                var code = btn.dataset.code;
                expandedCode = (expandedCode === code) ? null : code;
                renderDetailTable(recs);
            });
        });
    }

    function updateDetailTotal(recs) {
        var total = 0;
        recs.forEach(function (r) {
            total += r.line_total_excl_gst != null ? r.line_total_excl_gst : ((r.fee || r.fee_excl_gst || 0) * (r.quantity || 1));
        });
        document.querySelector("#detail-grand-total .grand-total-amount").textContent = "$" + total.toFixed(2);
    }

    // ── Back button ──
    document.getElementById("back-btn").addEventListener("click", function () {
        document.getElementById("job-detail-view").style.display = "none";
        document.getElementById("jobs-list-view").style.display = "";
        currentJobId = null;
        currentJobData = null;
        expandedCode = null;
    });

    // ── Job detail sub-tabs ──
    var jobTabs = document.querySelectorAll("[data-job-tab]");
    jobTabs.forEach(function (btn) {
        btn.addEventListener("click", function () {
            activateJobTab(btn.dataset.jobTab);
        });
    });

    function activateJobTab(tabId) {
        jobTabs.forEach(function (t) { t.classList.remove("active"); });
        document.querySelectorAll(".job-tab-content").forEach(function (s) { s.classList.remove("active"); });
        document.querySelector('[data-job-tab="' + tabId + '"]').classList.add("active");
        document.getElementById(tabId).classList.add("active");
    }

    // ── Init ──
    loadJobs();
})();
