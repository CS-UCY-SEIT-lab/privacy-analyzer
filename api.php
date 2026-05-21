<?php
header('Content-Type: application/json');

$allowed = ['predict', 'predict-file', 'analyze-stage2', 'health', 'save-feedback'];
$endpoint = $_GET['endpoint'] ?? '';

if (!in_array($endpoint, $allowed, true)) {
    http_response_code(400);
    echo json_encode(['error' => 'Invalid endpoint']);
    exit;
}

$baseDir = '/sys-data/WebData/projects/reqprivacyanalyzer';
$python = '/WebData/projects/reqprivacyanalyzer/run_py39.sh';
$bridge = $baseDir . '/bridge.py';
$hfCache = $baseDir . '/.hf_cache';
$stCache = $baseDir . '/.st_cache';

@mkdir($hfCache, 0755, true);
@mkdir($stCache, 0755, true);

$descriptorSpec = [
    0 => ["pipe", "r"],
    1 => ["pipe", "w"],
    2 => ["pipe", "w"]
];

$env = [
    'PATH' => '/usr/bin:/bin:/usr/local/bin',
    'HOME' => $baseDir,
    'VIRTUAL_ENV' => $baseDir . '/venv',
    'PYTHONPATH' =>
        $baseDir . '/venv/lib/python3.9/site-packages:' .
        $baseDir . '/venv/lib64/python3.9/site-packages',
    'LD_LIBRARY_PATH' => '/usr/lib64:/usr/lib:/lib64:/lib',
    'HF_HOME' => $hfCache,
    'HUGGINGFACE_HUB_CACHE' => $hfCache,
    'TRANSFORMERS_CACHE' => $hfCache,
    'SENTENCE_TRANSFORMERS_HOME' => $stCache,
    'PYTHONUNBUFFERED' => '1'
];

function run_python($cmd, $descriptorSpec, $baseDir, $env, $stdin = '') {
    $process = proc_open($cmd, $descriptorSpec, $pipes, $baseDir, $env);

    if (!is_resource($process)) {
        http_response_code(500);
        echo json_encode(['error' => 'Failed to start Python process']);
        exit;
    }

    fwrite($pipes[0], $stdin ?: '');
    fclose($pipes[0]);

    $stdout = stream_get_contents($pipes[1]);
    fclose($pipes[1]);

    $stderr = stream_get_contents($pipes[2]);
    fclose($pipes[2]);

    $exitCode = proc_close($process);

    if ($exitCode !== 0) {
        http_response_code(500);
        echo json_encode([
            'error' => 'Python bridge failed',
            'details' => $stderr ?: $stdout
        ]);
        exit;
    }

    echo $stdout;
    exit;
}

if ($endpoint === 'save-feedback') {
    $input = json_decode(file_get_contents("php://input"), true);

    $text = $input["text"] ?? "";
    $model_label = $input["model_label"] ?? "";
    $user_label = $input["user_label"] ?? "";

    if ($text === "" || $user_label === "") {
        http_response_code(400);
        echo json_encode(["error" => "Missing feedback data"]);
        exit;
    }

    $dataDir = $baseDir . "/data";
    @mkdir($dataDir, 0775, true);

    $file = $dataDir . "/privacy_feedback_dataset.csv";
    $isNew = !file_exists($file);
    $fp = fopen($file, "a");

    if (!$fp) {
        http_response_code(500);
        echo json_encode(["error" => "Could not open CSV file"]);
        exit;
    }

    if ($isNew) {
        fputcsv($fp, ["timestamp", "text", "model_label", "user_label", "was_corrected"]);
    }

    $was_corrected = ($model_label !== "" && strval($model_label) !== strval($user_label)) ? 1 : 0;

    fputcsv($fp, [
        date("Y-m-d H:i:s"),
        $text,
        $model_label,
        $user_label,
        $was_corrected
    ]);

    fclose($fp);

    echo json_encode(["status" => "saved"]);
    exit;
}

if ($endpoint === 'predict-file') {
    if (!isset($_FILES['file']) || !is_uploaded_file($_FILES['file']['tmp_name'])) {
        http_response_code(400);
        echo json_encode(['error' => 'No file uploaded']);
        exit;
    }

    $origName = $_FILES['file']['name'] ?? 'upload.bin';
    $ext = strtolower(pathinfo($origName, PATHINFO_EXTENSION));

    if (!in_array($ext, ['txt', 'csv', 'xlsx'], true)) {
        http_response_code(400);
        echo json_encode(['error' => 'Unsupported file type']);
        exit;
    }

    $safePath = sys_get_temp_dir() . '/privacy_upload_' . uniqid() . '.' . $ext;

    if (!move_uploaded_file($_FILES['file']['tmp_name'], $safePath)) {
        http_response_code(500);
        echo json_encode(['error' => 'Failed to store uploaded file']);
        exit;
    }

    $cmd = escapeshellarg($python) . ' ' . escapeshellarg($bridge) . ' predict-file ' . escapeshellarg($safePath);
    run_python($cmd, $descriptorSpec, $baseDir, $env);
}

$raw = file_get_contents('php://input');
$cmd = escapeshellarg($python) . ' ' . escapeshellarg($bridge) . ' ' . escapeshellarg($endpoint);

run_python($cmd, $descriptorSpec, $baseDir, $env, $raw);
?>