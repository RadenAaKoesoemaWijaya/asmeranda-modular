; ===========================================================================
; Asmeranda AI - Inno Setup Script
; Modular Machine Learning Platform Windows Installer
; ===========================================================================

[Setup]
AppName=Asmeranda AI
AppVersion=2.0.0
AppPublisher=PT. Asmer Sahabat Sukses
AppPublisherURL=https://www.asmer.co.id
AppSupportURL=https://www.asmer.co.id
DefaultDirName={autopf}\Asmeranda AI
DefaultGroupName=Asmeranda AI
OutputDir=.\InstallerOutput
OutputBaseFilename=AsmerandaAI_Setup_v2.0.0
Compression=lzma2/max
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64
PrivilegesRequired=admin
DisableProgramGroupPage=yes
UninstallDisplayIcon={app}\run_local.bat

[Languages]
Name: "indonesian"; MessagesFile: "compiler:Languages\Indonesian.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Buat pintasan di Desktop (Create Desktop Shortcut)"; GroupDescription: "Pintasan Tambahan:"; Flags: unchecked

[Files]
; Salin backend, frontend, core, dan konfigurasi modular
Source: "backend\*"; DestDir: "{app}\backend"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "frontend\*"; DestDir: "{app}\frontend"; Flags: ignoreversion recursesubdirs createallsubdirs; Excludes: "node_modules\*, .next\*, out\*"
Source: "core\*"; DestDir: "{app}\core"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "data\*"; DestDir: "{app}\data"; Flags: ignoreversion recursesubdirs createallsubdirs skipifsourcedoesntexist
Source: "nginx\*"; DestDir: "{app}\nginx"; Flags: ignoreversion recursesubdirs createallsubdirs skipifsourcedoesntexist
Source: "run_local.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "run_local.ps1"; DestDir: "{app}"; Flags: ignoreversion
Source: "start_docker.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "stop_docker.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "docker-compose.yml"; DestDir: "{app}"; Flags: ignoreversion
Source: "Dockerfile"; DestDir: "{app}"; Flags: ignoreversion
Source: "Dockerfile.backend"; DestDir: "{app}"; Flags: ignoreversion
Source: ".dockerignore"; DestDir: "{app}"; Flags: ignoreversion
Source: ".env.example"; DestDir: "{app}"; Flags: ignoreversion
Source: "workflow_validator.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "deploy-docker-desktop.ps1"; DestDir: "{app}"; Flags: ignoreversion
Source: "deploy-local.sh"; DestDir: "{app}"; Flags: ignoreversion
Source: "README.md"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\Jalankan Asmeranda AI (Localhost)"; Filename: "{app}\run_local.bat"
Name: "{group}\Jalankan Asmeranda AI (Docker)"; Filename: "{app}\start_docker.bat"
Name: "{group}\Hentikan Kontainer Docker"; Filename: "{app}\stop_docker.bat"
Name: "{group}\Uninstall Asmeranda AI"; Filename: "{uninstallexe}"
Name: "{autodesktop}\Asmeranda AI"; Filename: "{app}\run_local.bat"; Tasks: desktopicon

[Run]
Filename: "{app}\run_local.bat"; Description: "Jalankan Asmeranda AI sekarang"; Flags: nowait postinstall skipifsilent shellexec
