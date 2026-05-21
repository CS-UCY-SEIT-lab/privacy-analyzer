const isLocal =
    window.location.hostname === "127.0.0.1" ||
    window.location.hostname === "localhost" ||
    window.location.protocol === "file:";

const API_BASE = isLocal
    ? "http://127.0.0.1:5000"
    : `${window.location.origin}/privacyanalyzer/api.php`;

const API_PREDICT_URL = isLocal
    ? `${API_BASE}/predict`
    : `${API_BASE}?endpoint=predict`;

const API_FILE_URL = isLocal
    ? `${API_BASE}/predict-file`
    : `${API_BASE}?endpoint=predict-file`;

const API_STAGE2_URL = isLocal
    ? `${API_BASE}/analyze-stage2`
    : `${API_BASE}?endpoint=analyze-stage2`;

const API_FEEDBACK_URL = isLocal
    ? `${API_BASE}/save-feedback`
    : `${API_BASE}?endpoint=save-feedback`;

let LAST_RESULTS = [];
let PRIVACY_REVIEW_RESULTS = [];
let STAGE2_RESULTS = [];

const RIGHTS_INFO = {

    "gdpr-article 5": {
        right: "Principles relating to processing of personal data",
        description: "Personal data must be processed lawfully, fairly, and transparently, collected for specified and legitimate purposes, limited to what is necessary, kept accurate, stored only as long as needed, and protected with appropriate security.",
        article: "GDPR Article 5"
    },

    "gdpr-article 6": {
        right: "Lawfulness of processing",
        description: "Processing is lawful only if at least one valid legal basis applies, such as consent, contract, legal obligation, vital interests, public task, or legitimate interests.",
        article: "GDPR Article 6"
    },

    "gdpr-article 7": {
        right: "Conditions for consent",
        description: "Consent must be freely given, specific, informed, and clearly expressed through an affirmative action, and users must be able to withdraw it easily.",
        article: "GDPR Article 7"
    },

    "gdpr-article 12": {
        right: "Transparent information, communication and modalities for the exercise of the rights of the data subject",
        description: "Information about processing and the exercise of user rights must be provided in a concise, transparent, intelligible, and easily accessible form, usually free of charge and within the required time limits.",
        article: "GDPR Article 12"
    },

    "gdpr-article 13": {
        right: "Information to be provided where personal data are collected from the data subject",
        description: "When personal data is collected directly from users, they must be informed about the controller, the purposes and legal basis of processing, recipients, retention, and their rights.",
        article: "GDPR Article 13"
    },

    "gdpr-article 14": {
        right: "Information to be provided where personal data have not been obtained from the data subject",
        description: "When personal data is obtained indirectly, users must be informed about the source of the data, the purpose of processing, the categories of data involved, retention, and their rights.",
        article: "GDPR Article 14"
    },

    "gdpr-article 15": {
        right: "Right of access by the data subject",
        description: "Users have the right to obtain confirmation that their personal data is being processed, access that data, and receive information about how and why it is processed.",
        article: "GDPR Article 15"
    },

    "gdpr-article 16": {
        right: "Right to rectification",
        description: "Users have the right to correct inaccurate personal data and complete incomplete personal data.",
        article: "GDPR Article 16"
    },

    "gdpr-article 17": {
        right: "Right to erasure ('right to be forgotten')",
        description: "Users have the right to request deletion of their personal data in specific circumstances, such as when the data is no longer needed, consent is withdrawn, or the processing is unlawful.",
        article: "GDPR Article 17"
    },

    "gdpr-article 18": {
        right: "Right to restriction of processing",
        description: "Users can request that the processing of their personal data be restricted in certain situations, for example while the accuracy or lawfulness of the data is being assessed.",
        article: "GDPR Article 18"
    },

    "gdpr-article 19": {
        right: "Notification obligation regarding rectification or erasure of personal data or restriction of processing",
        description: "When personal data is corrected, erased, or its processing is restricted, the controller must communicate this to recipients of the data unless this is impossible or involves disproportionate effort.",
        article: "GDPR Article 19"
    },

    "gdpr-article 20": {
        right: "Right to data portability",
        description: "Users can receive personal data they provided in a structured, commonly used, machine-readable format and, where possible, have it transmitted to another controller.",
        article: "GDPR Article 20"
    },

    "gdpr-article 21": {
        right: "Right to object",
        description: "Users have the right to object to certain types of processing, especially direct marketing, and in some cases processing must stop unless compelling legitimate grounds exist.",
        article: "GDPR Article 21"
    },

    "gdpr-article 22": {
        right: "Automated individual decision-making, including profiling",
        description: "Users have the right not to be subject to decisions based solely on automated processing, including profiling, when those decisions significantly affect them, subject to limited exceptions.",
        article: "GDPR Article 22"
    },

    "gdpr-article 25": {
        right: "Data protection by design and by default",
        description: "Systems must include appropriate privacy safeguards from the design stage and ensure that, by default, only personal data necessary for each purpose is processed.",
        article: "GDPR Article 25"
    },

    "gdpr-article 29": {
        right: "Processing under the authority of the controller or processor",
        description: "Persons acting under the authority of a controller or processor may process personal data only on the controller’s instructions, unless required by law.",
        article: "GDPR Article 29"
    },

    "gdpr-article 30": {
        right: "Records of processing activities",
        description: "Controllers and processors must maintain records describing their processing activities, including purposes, categories of data, recipients, retention, and security measures where applicable.",
        article: "GDPR Article 30"
    },

    "gdpr-article 31": {
        right: "Cooperation with the supervisory authority",
        description: "Controllers and processors must cooperate with the supervisory authority when requested in the performance of its tasks.",
        article: "GDPR Article 31"
    },

    "gdpr-article 32": {
        right: "Security of processing",
        description: "Appropriate technical and organizational measures must be implemented to ensure a level of security appropriate to the risk, such as confidentiality, integrity, encryption, and regular testing.",
        article: "GDPR Article 32"
    },

    "gdpr-article 33": {
        right: "Notification of a personal data breach to the supervisory authority",
        description: "A personal data breach must be notified to the supervisory authority without undue delay and, where feasible, within 72 hours, unless the breach is unlikely to result in a risk to individuals.",
        article: "GDPR Article 33"
    },

    "gdpr-article 34": {
        right: "Communication of a personal data breach to the data subject",
        description: "When a personal data breach is likely to result in a high risk to individuals, affected users must be informed without undue delay in clear and understandable language.",
        article: "GDPR Article 34"
    },

    "ccpa-right to know": {
        right: "Right to Know",
        description: "Consumers can request that a business disclose what personal information it has collected about them, including categories, specific pieces of information, sources, purposes, and categories of third parties to whom the information is disclosed, sold, or shared.",
        article: "CCPA Right to Know"
    },

    "ccpa-right to access": {
        right: "Right to Access",
        description: "Consumers can access the personal information a business has collected about them as part of their right to know, including details about its collection, use, disclosure, sale, or sharing.",
        article: "CCPA Right to Access"
    },

    "ccpa-right to delete": {
        right: "Right to Delete",
        description: "Consumers can request deletion of personal information collected from them, and the business must also direct relevant service providers or contractors to delete it, subject to statutory exceptions.",
        article: "CCPA Right to Delete"
    },

    "ccpa-right to correct": {
        right: "Right to Correct",
        description: "Consumers can request correction of inaccurate personal information maintained by a business, taking into account the nature of the information and the purposes of processing.",
        article: "CCPA Right to Correct"
    },

    "ccpa-right to opt-out of sale or sharing": {
        right: "Right to Opt-Out of Sale or Sharing",
        description: "Consumers can direct a business not to sell their personal information and not to share it for cross-context behavioral advertising, and businesses must provide a clear method for exercising this choice.",
        article: "CCPA Right to Opt-Out of Sale/Sharing"
    },

    "ccpa-right to non-discrimination": {
        right: "Right to Non-Discrimination",
        description: "Consumers must not be discriminated against for exercising their CCPA rights, including through denial of goods or services, different prices, different levels of quality, or other retaliatory treatment, except where permitted by law.",
        article: "CCPA Right to Non-Discrimination"
    },

    "ccpa-notice at collection": {
        right: "Notice at Collection",
        description: "Consumers must be informed, at or before the point of collection, about the categories of personal information collected and the purposes for which that information will be used.",
        article: "CCPA Notice at Collection"
    },

    "ccpa-right to limit use and disclosure of sensitive personal information": {
        right: "Right to Limit Use and Disclosure of Sensitive Personal Information",
        description: "Consumers can direct a business to limit the use and disclosure of sensitive personal information to what is necessary to perform services or provide goods reasonably expected by an average consumer, or for other permitted purposes defined by law.",
        article: "CCPA Right to Limit Use/Disclosure of Sensitive Personal Information"
    }
};

function setPrivacyReview(html) {
    const el = document.getElementById("privacyReviewContainer");
    if (el) el.innerHTML = html;
}

function setLegalResults(html) {
    const el = document.getElementById("legalResultsContainer");
    if (el) el.innerHTML = html;
}

function escapeHtml(text) {
    if (text === null || text === undefined) return "";
    return String(text)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function parseRequirements(rawText) {
    return rawText
        .split(/\n+/)
        .map(t => t.trim())
        .filter(Boolean);
}

function getDisplayLabel(label) {
    const map = {
        "gdpr-article 5": "Article 5",
        "gdpr-article 6": "Article 6",
        "gdpr-article 7": "Article 7",
        "gdpr-article 12": "Article 12",
        "gdpr-article 13": "Article 13",
        "gdpr-article 14": "Article 14",
        "gdpr-article 15": "Article 15",
        "gdpr-article 16": "Article 16",
        "gdpr-article 17": "Article 17",
        "gdpr-article 18": "Article 18",
        "gdpr-article 19": "Article 19",
        "gdpr-article 20": "Article 20",
        "gdpr-article 21": "Article 21",
        "gdpr-article 22": "Article 22",
        "gdpr-article 25": "Article 25",
        "gdpr-article 29": "Article 29",
        "gdpr-article 30": "Article 30",
        "gdpr-article 31": "Article 31",
        "gdpr-article 32": "Article 32",
        "gdpr-article 33": "Article 33",
        "gdpr-article 34": "Article 34",

        "ccpa-right to know": "Right to Know",
        "ccpa-right to access": "Right to Access",
        "ccpa-right to delete": "Right to Delete",
        "ccpa-right to correct": "Right to Correct",
        "ccpa-right to opt-out of sale or sharing": "Right to Opt-Out of Sale/Sharing",
        "ccpa-right to opt-out": "Right to Opt-Out of Sale/Sharing",
        "ccpa-right to non-discrimination": "Right to Non-Discrimination",
        "ccpa-non-discrimination": "Right to Non-Discrimination",
        "ccpa-notice at collection": "Notice at Collection",
        "ccpa-notice at collection & non-discrimination": "Notice at Collection / Non-Discrimination",
        "ccpa-right to limit use and disclosure of sensitive personal information":
            "Right to Limit Use and Disclosure of Sensitive Personal Information",
        "ccpa-right to limit spi":
            "Right to Limit Use and Disclosure of Sensitive Personal Information"
    };

    return map[label] || label;
}

function getRightInfo(label) {
    return RIGHTS_INFO[label] || {
        right: getDisplayLabel(label),
        description: "No description available.",
        article: getDisplayLabel(label)
    };
}

function getMatchColorClass(label) {
    const l = (label || "").toLowerCase();

    if (l === "__right_to_information__") {
        return { chip: "match-green", highlight: "hl-green" };
    }
    if (l.includes("delete") || l.includes("article 17")) {
        return { chip: "match-blue", highlight: "hl-blue" };
    }
    if (l.includes("article 15") || l.includes("know") || l.includes("access")) {
        return { chip: "match-green", highlight: "hl-green" };
    }
    if (l.includes("article 16") || l.includes("correct") || l.includes("rectif")) {
        return { chip: "match-purple", highlight: "hl-purple" };
    }
    if (l.includes("article 20") || l.includes("portability")) {
        return { chip: "match-orange", highlight: "hl-orange" };
    }
    if (l.includes("article 21") || l.includes("object")) {
        return { chip: "match-red", highlight: "hl-red" };
    }
    if (l.includes("article 22") || l.includes("automated")) {
        return { chip: "match-pink", highlight: "hl-pink" };
    }
    if (l.includes("article 32") || l.includes("security")) {
        return { chip: "match-teal", highlight: "hl-teal" };
    }
    if (l.includes("article 33") || l.includes("article 34") || l.includes("breach")) {
        return { chip: "match-yellow", highlight: "hl-yellow" };
    }

    return { chip: "match-default", highlight: "hl-default" };
}

function buildDisplayMatches(matches) {
    if (!Array.isArray(matches)) return [];

    const uniqueMatches = [...new Set(matches.map(m => String(m).toLowerCase().trim()))];
    const displayMatches = [];
    const used = new Set();

    const groups = [
        {
            groupLabel: "__access_group__",
            groupTitle: "Right to Access",
            labels: ["gdpr-article 15", "ccpa-right to access"]
        },
        {
            groupLabel: "__delete_group__",
            groupTitle: "Right to Deletion / Erasure",
            labels: ["gdpr-article 17", "ccpa-right to delete"]
        },
        {
            groupLabel: "__correction_group__",
            groupTitle: "Right to Correction / Rectification",
            labels: ["gdpr-article 16", "ccpa-right to correct"]
        },
        {
            groupLabel: "__information_group__",
            groupTitle: "Right to Information / Notice",
            labels: ["gdpr-article 13", "gdpr-article 14", "ccpa-notice at collection"]
        },
        {
            groupLabel: "__optout_group__",
            groupTitle: "Right to Object / Opt-Out",
            labels: ["gdpr-article 21", "ccpa-right to opt-out", "ccpa-right to opt-out of sale or sharing"]
        }
    ];

    groups.forEach(group => {
        const matchedLabels = group.labels.filter(label => uniqueMatches.includes(label));

        if (matchedLabels.length > 1) {
            displayMatches.push({
                kind: "group",
                label: group.groupLabel,
                right: group.groupTitle,
                children: matchedLabels.map(label => {
                    const info = getRightInfo(label);
                    used.add(label);

                    return {
                        kind: "standard",
                        label: label,
                        right: info.right,
                        description: info.description,
                        article: info.article
                    };
                })
            });
        }
    });

    uniqueMatches.forEach(match => {
        if (used.has(match)) return;

        const info = getRightInfo(match);
        displayMatches.push({
            kind: "standard",
            label: match,
            right: info.right,
            description: info.description,
            article: info.article
        });
    });

    return displayMatches;
}

function renderMatchChips(matches) {
    if (!Array.isArray(matches) || matches.length === 0) {
        return `
            <div class="message warning inline-msg">
                No matches detected.
            </div>
        `;
    }

    const displayMatches = buildDisplayMatches(matches);

    return displayMatches.map(item => {
        const color = getMatchColorClass(item.label);

        if (item.kind === "group") {
            return `
                <div class="match-card ${color.chip} grouped-match-card">
                    <div class="match-card-title">${escapeHtml(item.right)}</div>

                    <div class="grouped-subcards">
                        ${item.children.map(child => {
                            const childColor = getMatchColorClass(child.label);

                            return `
                                <div class="match-card ${childColor.chip} match-subcard">
                                    <div class="match-card-title">${escapeHtml(child.right)}</div>
                                    <div class="match-card-desc">${escapeHtml(child.description)}</div>
                                    <div class="match-card-article">(${escapeHtml(child.article)})</div>
                                </div>
                            `;
                        }).join("")}
                    </div>
                </div>
            `;
        }

        return `
            <div class="match-card ${color.chip}">
                <div class="match-card-title">${escapeHtml(item.right)}</div>
                <div class="match-card-desc">${escapeHtml(item.description)}</div>
                <div class="match-card-article">(${escapeHtml(item.article)})</div>
            </div>
        `;
    }).join("");
}

function normalizeResultItem(item) {
    if (!item) return null;

    const stage1 = item.stage1 || {};
    const modelPrediction = stage1.model_prediction || {};
    const finalDecision = stage1.final_decision || {};
    const fallbackLabel = Number(finalDecision.label ?? modelPrediction.label ?? 0);

    return {
        input_text: item.input_text || "",
        raw: item,
        isSelectedPrivacy: fallbackLabel === 1
    };
}

async function saveFeedbackCorrection(item) {
    if (!item || !item.raw || !item.raw.stage1) return;

    const modelLabel = item.raw.stage1.model_prediction?.label;
    const userLabel = item.isSelectedPrivacy ? 1 : 0;

    try {
        await fetch(API_FEEDBACK_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                text: item.input_text,
                model_label: modelLabel,
                user_label: userLabel
            })
        });
    } catch (error) {
        console.error("Feedback save error:", error);
    }
}

function getSelectedPrivacyCount() {
    return PRIVACY_REVIEW_RESULTS.filter(item => item.isSelectedPrivacy).length;
}

function getNonPrivacyCount() {
    return PRIVACY_REVIEW_RESULTS.filter(item => !item.isSelectedPrivacy).length;
}

function renderPrivacyReviewSection() {
    if (!PRIVACY_REVIEW_RESULTS.length) {
        return `
            <div class="message info">
                Detected requirements will appear here for review.
            </div>
        `;
    }

    const privacyCount = getSelectedPrivacyCount();
    const nonPrivacyCount = getNonPrivacyCount();

    return `
        <div class="privacy-review-list">
            ${PRIVACY_REVIEW_RESULTS.map((item, index) => `
                <div class="privacy-review-item">
                    <label class="privacy-review-checkbox">
                        <input
                            type="checkbox"
                            class="privacy-toggle"
                            data-index="${index}"
                            ${item.isSelectedPrivacy ? "checked" : ""}
                        >
                        <span class="privacy-toggle-custom"></span>
                    </label>

                    <div class="privacy-review-content">
                        <div class="privacy-review-line">
                            <div class="privacy-review-text">
                                ${index + 1}. ${escapeHtml(item.input_text || "")}
                            </div>
                            <span class="privacy-review-status ${item.isSelectedPrivacy ? "is-privacy" : "is-nonprivacy"}">
                                ${item.isSelectedPrivacy ? "Privacy-related" : "Not selected"}
                            </span>
                        </div>
                    </div>
                </div>
            `).join("")}
        </div>

        <div class="privacy-review-summary" style="margin-top: 16px;">
            <div class="privacy-count-chip privacy-count-positive">
                Selected: ${privacyCount}
            </div>
            <div class="privacy-count-chip privacy-count-neutral">
                Unselected: ${nonPrivacyCount}
            </div>
        </div>

        <div class="privacy-review-actions">
            <button type="button" id="runStage2Btn" class="btn btn-primary">
                Continue to Legal Mapping
            </button>
        </div>
    `;
}

function renderStage2ResultsSection() {
    if (!STAGE2_RESULTS.length) {
        return `
            <div class="message info">
                Legal mapping results will appear here after review.
            </div>
        `;
    }

    return STAGE2_RESULTS.map((item, index) => {
        const stage2 = item.stage2 || {};
        const matches = Array.isArray(stage2.matches) ? [...new Set(stage2.matches)] : [];

        return `
            <div class="result-card">
                <div class="result-title">Requirement ${index + 1} – Legal Mapping</div>
                <div class="match-label">Mapped Legal Concepts</div>
                <div class="matches-wrap matches-wrap-detailed">
                    ${renderMatchChips(matches)}
                </div>
                <div class="result-text" style="margin-top: 16px;">
                    ${escapeHtml(item.input_text || "")}
                </div>
            </div>
        `;
    }).join("");
}

function renderAllSections() {
    const exportWrap = document.getElementById("exportWrap");
    if (exportWrap) {
        exportWrap.style.display = STAGE2_RESULTS.length > 0 ? "block" : "none";
    }

    setPrivacyReview(renderPrivacyReviewSection());
    setLegalResults(renderStage2ResultsSection());

    bindDynamicEvents();
}

function renderStage1Results(results) {
    PRIVACY_REVIEW_RESULTS = Array.isArray(results)
        ? results.map(normalizeResultItem).filter(Boolean)
        : [];

    STAGE2_RESULTS = [];
    LAST_RESULTS = results || [];

    renderAllSections();
}

async function findMatches() {
    const textarea = document.getElementById("requirementsInput");
    const matchBtn = document.getElementById("matchBtn");

    if (!textarea || !matchBtn) return;

    const rawText = textarea.value.trim();
    const texts = parseRequirements(rawText);

    if (texts.length === 0) {
        setPrivacyReview(`
            <div class="message error">
                Please enter at least one requirement.
            </div>
        `);
        setLegalResults(`
            <div class="message info">
                Legal mapping results will appear here after review.
            </div>
        `);
        textarea.focus();
        return;
    }

    matchBtn.disabled = true;
    matchBtn.textContent = "Checking...";

    setPrivacyReview(`
        <div class="message loading">
            Running privacy classification...
        </div>
    `);

    setLegalResults(`
        <div class="message info">
            Legal mapping results will appear here after review.
        </div>
    `);

    try {
        const payload = texts.length === 1
            ? { text: texts[0], proceed_to_stage2: false }
            : { texts: texts, proceed_to_stage2: false };

        const response = await fetch(API_PREDICT_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            let errorText = `HTTP ${response.status}`;
            try {
                const errData = await response.json();
                if (errData.error) {
                    errorText = `${response.status} - ${errData.error}`;
                }
            } catch {
                errorText = `${response.status} - Request failed`;
            }

            setPrivacyReview(`
                <div class="message error">
                    Request failed: ${escapeHtml(errorText)}
                </div>
            `);
            setLegalResults(`
                <div class="message info">
                    Legal mapping results will appear here after review.
                </div>
            `);
            return;
        }

        const data = await response.json();
        const results = Array.isArray(data.results) ? data.results : [data];
        renderStage1Results(results);

    } catch (error) {
        console.error("Match error:", error);
        setPrivacyReview(`
            <div class="message error">
                Could not connect to the backend API. Please make sure the backend service is running and reachable.
            </div>
        `);
        setLegalResults(`
            <div class="message info">
                Legal mapping results will appear here after review.
            </div>
        `);
    } finally {
        matchBtn.disabled = false;
        matchBtn.textContent = "Find Matches";
    }
}

async function uploadFileAndMatch() {
    const fileInput = document.getElementById("fileInput");
    const uploadBtn = document.getElementById("uploadBtn");

    if (!fileInput) return;

    if (!fileInput.files || fileInput.files.length === 0) {
        setPrivacyReview(`
            <div class="message error">
                Please choose a file first.
            </div>
        `);
        setLegalResults(`
            <div class="message info">
                Legal mapping results will appear here after review.
            </div>
        `);
        return;
    }

    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append("file", file);

    if (uploadBtn) {
        uploadBtn.disabled = true;
        uploadBtn.textContent = "Uploading...";
    }

    setPrivacyReview(`
        <div class="message loading">
            Uploading file and running privacy classification...
        </div>
    `);

    setLegalResults(`
        <div class="message info">
            Legal mapping results will appear here after review.
        </div>
    `);

    try {
        const response = await fetch(API_FILE_URL, {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            let errorText = `HTTP ${response.status}`;
            try {
                const errData = await response.json();
                if (errData.error) {
                    errorText = `${response.status} - ${errData.error}`;
                }
            } catch {
                errorText = `${response.status} - Upload failed`;
            }

            setPrivacyReview(`
                <div class="message error">
                    Upload failed: ${escapeHtml(errorText)}
                </div>
            `);
            setLegalResults(`
                <div class="message info">
                    Legal mapping results will appear here after review.
                </div>
            `);
            return;
        }

        const data = await response.json();
        renderStage1Results(data.results || []);

    } catch (error) {
        console.error("Upload error:", error);
        setPrivacyReview(`
            <div class="message error">
                Could not upload file or connect to the backend API.
            </div>
        `);
        setLegalResults(`
            <div class="message info">
                Legal mapping results will appear here after review.
            </div>
        `);
    } finally {
        if (uploadBtn) {
            uploadBtn.disabled = false;
            uploadBtn.textContent = "Upload and Match";
        }
    }
}

async function runStage2ForSelected() {
    const selectedItems = PRIVACY_REVIEW_RESULTS
        .map((item, index) => ({ item, index }))
        .filter(entry => entry.item.isSelectedPrivacy);

    if (!selectedItems.length) {
        setPrivacyReview(renderPrivacyReviewSection());
        setLegalResults(`
            <div class="message warning">
                Please select at least one privacy-related requirement.
            </div>
        `);
        bindDynamicEvents();
        return;
    }

    const runBtn = document.getElementById("runStage2Btn");
    if (runBtn) {
        runBtn.disabled = true;
        runBtn.textContent = "Running Mapping...";
    }

    STAGE2_RESULTS = [];
    renderAllSections();

    setLegalResults(`
        <div class="message loading">
            Running legal mapping...
        </div>
    `);

    try {
        const responses = await Promise.all(
            selectedItems.map(async ({ item }) => {
                const response = await fetch(API_STAGE2_URL, {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({
                        text: item.input_text,
                        override_label: 1,
                        force_stage2: false,
                        decompose: true
                    })
                });

                if (!response.ok) {
                    let errorText = `HTTP ${response.status}`;
                    try {
                        const errData = await response.json();
                        if (errData.error) {
                            errorText = `${response.status} - ${errData.error}`;
                        }
                    } catch {
                        errorText = `${response.status} - Stage 2 request failed`;
                    }
                    throw new Error(errorText);
                }

                return await response.json();
            })
        );

        STAGE2_RESULTS = responses;
        renderAllSections();

    } catch (error) {
        console.error("Stage 2 error:", error);
        STAGE2_RESULTS = [];
        setLegalResults(`
            <div class="message error">
                Could not run legal mapping: ${escapeHtml(error.message || "Unknown error")}
            </div>
        `);
        bindDynamicEvents();
    }
}

function bindDynamicEvents() {
    document.querySelectorAll(".privacy-toggle").forEach(toggle => {
        toggle.addEventListener("change", async function (event) {
            event.preventDefault();
            event.stopPropagation();

            const index = Number(this.dataset.index);
            if (Number.isNaN(index) || !PRIVACY_REVIEW_RESULTS[index]) return;

            PRIVACY_REVIEW_RESULTS[index].isSelectedPrivacy = this.checked;

            await saveFeedbackCorrection(PRIVACY_REVIEW_RESULTS[index]);

            renderAllSections();
        });
    });

    const runStage2Btn = document.getElementById("runStage2Btn");
    if (runStage2Btn) {
        runStage2Btn.addEventListener("click", function (event) {
            event.preventDefault();
            event.stopPropagation();
            runStage2ForSelected();
        });
    }
}

function exportResultsCSV() {
    if (!STAGE2_RESULTS || STAGE2_RESULTS.length === 0) {
        alert("No legal mapping results to export.");
        return;
    }

    let csv = "Requirement,Detected Matches,Match Count,Evidence\n";

    STAGE2_RESULTS.forEach((item) => {
        const stage2 = item.stage2 || {};
        const requirement = String(item.input_text || "").replace(/"/g, '""');

        const matches = Array.isArray(stage2.matches)
            ? [...new Set(stage2.matches)].join(" | ").replace(/"/g, '""')
            : "";

        const matchCount = Array.isArray(stage2.matches)
            ? [...new Set(stage2.matches)].length
            : 0;

        const evidenceObj = stage2.evidence || {};
        const evidenceText = Object.entries(evidenceObj)
            .map(([label, phrases]) => `${label}: ${(phrases || []).join(" / ")}`)
            .join(" || ")
            .replace(/"/g, '""');

        csv += `"${requirement}","${matches}","${matchCount}","${evidenceText}"\n`;
    });

    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);

    const link = document.createElement("a");
    link.href = url;
    link.download = "privacy_matches.csv";

    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    URL.revokeObjectURL(url);
}

function clearAll() {
    const textarea = document.getElementById("requirementsInput");
    const fileInput = document.getElementById("fileInput");
    const selectedFile = document.getElementById("selectedFile");
    const exportWrap = document.getElementById("exportWrap");

    if (textarea) textarea.value = "";
    if (fileInput) fileInput.value = "";
    if (selectedFile) selectedFile.textContent = "";
    if (exportWrap) exportWrap.style.display = "none";

    LAST_RESULTS = [];
    PRIVACY_REVIEW_RESULTS = [];
    STAGE2_RESULTS = [];

    setPrivacyReview(`
        <div class="message info">
            Detected requirements will appear here for review.
        </div>
    `);

    setLegalResults(`
        <div class="message info">
            Legal mapping results will appear here after review.
        </div>
    `);

    if (textarea) textarea.focus();
}

function setupEvents() {

    document.querySelectorAll("form").forEach(form => {
        form.addEventListener("submit", function (event) {
            event.preventDefault();
        });
    });

    const textarea = document.getElementById("requirementsInput");
    const matchBtn = document.getElementById("matchBtn");
    const uploadBtn = document.getElementById("uploadBtn");
    const clearBtn = document.getElementById("clearBtn");
    const exportBtn = document.getElementById("exportBtn");
    const fileInput = document.getElementById("fileInput");
    const selectedFile = document.getElementById("selectedFile");

    if (matchBtn) {
        matchBtn.addEventListener("click", findMatches);
    }

    if (uploadBtn && fileInput) {
        uploadBtn.addEventListener("click", () => {
            fileInput.click();
        });
    }

    if (clearBtn) {
        clearBtn.addEventListener("click", clearAll);
    }

    if (exportBtn) {
        exportBtn.addEventListener("click", exportResultsCSV);
    }

    if (fileInput) {
        fileInput.addEventListener("change", function () {
            if (selectedFile) {
                if (fileInput.files && fileInput.files.length > 0) {
                    selectedFile.textContent = `Selected file: ${fileInput.files[0].name}`;
                    uploadFileAndMatch();
                } else {
                    selectedFile.textContent = "";
                }
            }
        });
    }

    if (textarea) {
        textarea.addEventListener("keydown", function (event) {
            if (event.ctrlKey && event.key === "Enter") {
                event.preventDefault(); // 🔴 extra safety
                findMatches();
            }
        });
    }

    clearAll();
}

document.addEventListener("DOMContentLoaded", setupEvents);