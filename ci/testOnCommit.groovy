pipeline {
    agent none
    options {
        disableConcurrentBuilds()
        timestamps()
        timeout(time: 120, unit: 'MINUTES')
    }
    parameters {
        string(name: 'LINUX_AGENT', defaultValue: 'ovmsbuilder', description: 'Linux agent with Docker')
        string(name: 'WINDOWS_AGENT', defaultValue: 'win_ovms', description: 'Windows x64 agent with Visual Studio 2022, CMake and Git')
    }
    stages {
        stage('Build and Test') {
            parallel {
                stage('Linux') {
                    agent { label "${params.LINUX_AGENT}" }
                    stages {
                        stage('Configure') {
                            steps {
                                checkout scm
                                script {
                                    env.SHORT_COMMIT = sh(returnStdout: true, script: 'git rev-parse --short HEAD').trim()
                                }
                            }
                        }
                        stage('Build Image') {
                            steps {
                                sh 'make docker_build OVMS_MEDIA_IMAGE_TAG=$SHORT_COMMIT'
                            }
                        }
                        stage('Build Demos Image') {
                            steps {
                                sh 'make ovms_demos_image OVMS_MEDIA_IMAGE_TAG=$SHORT_COMMIT'
                            }
                        }
                        stage('Unit Tests') {
                            steps {
                                sh 'make tests_in_docker OVMS_MEDIA_IMAGE_TAG=$SHORT_COMMIT'
                            }
                        }
                        stage('Test C++ Demos') {
                            steps {
                                sh 'make run_demos_in_docker OVMS_MEDIA_IMAGE_TAG=$SHORT_COMMIT'
                            }
                        }
                    }
                    post {
                        always {
                            junit testResults: 'test-results/**/test.xml', allowEmptyResults: true
                            archiveArtifacts artifacts: 'test-results/**/test.log', allowEmptyArchive: true
                        }
                    }
                }
                stage('Windows') {
                    agent { label "${params.WINDOWS_AGENT}" }
                    steps {
                        checkout scm
                        bat '''
                            call prepare_build_dependencies.bat
                            if errorlevel 1 exit /b 1
                            C:\\opt\\msys64\\usr\\bin\\make.exe build tests example-tests
                            set test_status=%ERRORLEVEL%
                            for /f "delims=" %%d in ('bazel --output_user_root=C:/b info bazel-testlogs') do xcopy /E /I /Y "%%d" test-results
                            exit /b %test_status%
                        '''
                    }
                    post {
                        always {
                            junit testResults: 'test-results/**/test.xml', allowEmptyResults: true
                            archiveArtifacts artifacts: 'test-results/**/test.log', allowEmptyArchive: true
                        }
                    }
                }
            }
        }
    }
}