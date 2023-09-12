pipeline {
    agent {
      label 'ovmscheck'
    }
    stages {
        stage('Configure') {
          steps {
            script {
              checkout scm
              shortCommit = sh(returnStdout: true, script: "git log -n 1 --pretty=format:'%h'").trim()
              echo shortCommit
            }
          }     
        }   
        stage("Run tests on commit") {
          steps {
              sh """
              env
              """
              echo shortCommit
              build job: "ovmsc/users/rasapala/OpenVinoMediapipe", parameters: [[$class: 'StringParameterValue', name: 'MEDIACOMMIT', value: shortCommit]]
          }    
        }
    }
}

