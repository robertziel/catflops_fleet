// CI: every gate in deploy/ci/gates.sh, each reported to GitHub as its own commit status (`ci/lb`, `ci/power`,
// `ci/usage-mcp`, `ci/deploy`, `ci/browser`) whose description is the suite's own count line.
//
// Each status's target_url is this build (env.BUILD_URL, from the Jenkins root URL setting), so a status leads to its
// log; a server that denies anonymous access opens it for its own accounts only.
//
// Requires on the Jenkins side:
//   - a Secret Text credential `github-pat`: a token allowed to write commit statuses on this repository
//   - a multibranch job over a plain Git source (not the GitHub source, which would post a second status of its own),
//     branches only: a fork's code never runs on a node that holds the Docker socket
//   - a node labelled `catflops-ci`: a host whose Docker daemon the gates drive (its socket); the label is this
//     project's, given by the deployment to whichever nodes may run it
//
// Everything real lives in deploy/ci/gates.sh, so any gate reproduces anywhere Docker is.

def GATES = ['lb', 'power', 'usage-mcp', 'deploy', 'browser']

pipeline {
  agent { label 'catflops-ci' }

  options {
    timestamps()
    disableConcurrentBuilds()
    timeout(time: 45, unit: 'MINUTES')
    buildDiscarder(logRotator(numToKeepStr: '30'))
  }

  environment {
    // The CI box serves other work too; each test container gets this many CPUs.
    CI_CPUS = '8'
  }

  stages {
    stage('Checkout') {
      steps {
        checkout scm
        script {
          env.GIT_SHA = sh(script: 'git rev-parse HEAD', returnStdout: true).trim()
          def origin = sh(script: 'git config --get remote.origin.url', returnStdout: true).trim()
          def repo = githubRepo(origin)
          if (!repo) { error("origin ${origin} is not a GitHub repository; no status can be posted") }
          env.GH_REPO = repo
          echo "Building ${env.GH_REPO}@${env.GIT_SHA}"
          GATES.each { notifyGitHub("ci/${it}", 'pending', 'running') }
        }
      }
    }

    stage('Gates') {
      steps {
        script {
          // Sequential: one gate's containers at a time on a shared box. A red gate does not stop the rest.
          GATES.each { g ->
            stage(g) {
              def rc = sh(returnStatus: true, script: "#!/bin/bash\nset -o pipefail; deploy/ci/gates.sh ${g} 2>&1 | tee .ci-${g}.log")
              def summary = sh(returnStdout: true,
                script: "grep -E '(passed|failed|error|OK|FAILED)' .ci-${g}.log | tail -1 | tr -s '=' ' ' | cut -c1-120 || true").trim()
              sh "rm -f .ci-${g}.log"
              if (rc == 0) {
                notifyGitHub("ci/${g}", 'success', summary ?: 'passed')
              } else {
                notifyGitHub("ci/${g}", 'failure', summary ?: "exit ${rc}")
                currentBuild.result = 'FAILURE'
              }
            }
          }
        }
      }
    }
  }

  post {
    aborted {
      script { GATES.each { notifyGitHub("ci/${it}", 'error', 'aborted') } }
    }
    cleanup { cleanWs(disableDeferredWipeout: true, deleteDirs: true, notFailBuild: true) }
  }
}

// owner/name from a GitHub remote URL, or null. @NonCPS: a regex Matcher cannot be saved with the pipeline state.
@NonCPS
String githubRepo(String url) {
  def m = (url =~ /github\.com[:\/]([^\/]+\/[^\/]+?)(\.git)?$/)
  return m.find() ? m.group(1) : null
}

// A commit status through the REST API, linking this build (see the header).
void notifyGitHub(String context, String state, String description) {
  if (!env.GIT_SHA || !env.GH_REPO) {
    echo "No commit or repository known yet; status ${context}=${state} not posted"
    return
  }
  withCredentials([string(credentialsId: 'github-pat', variable: 'GH_TOKEN')]) {
    writeFile file: '.github-status.json', text: groovy.json.JsonOutput.toJson([
      state: state, description: description, context: context, target_url: env.BUILD_URL
    ])
    def code = sh(returnStdout: true, script: '''
      set +x
      curl -sS -o /dev/null -w '%{http_code}' -X POST \
        -H "Authorization: Bearer $GH_TOKEN" -H "Accept: application/vnd.github+json" \
        --data @.github-status.json \
        "https://api.github.com/repos/''' + env.GH_REPO + '''/statuses/''' + env.GIT_SHA + '''"
    ''').trim()
    sh 'rm -f .github-status.json'
    if (code != '201') { error("GitHub refused status ${context}=${state} (HTTP ${code}); check the github-pat credential") }
    echo "status ${context}=${state} posted for ${env.GIT_SHA.take(8)}"
  }
}
