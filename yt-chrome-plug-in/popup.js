document.addEventListener("DOMContentLoaded", async () => {

  const outputDiv = document.getElementById("output");

  // LOCAL TESTING ONLY
  // Do NOT commit your real API key to GitHub.
  const API_KEY = "YOUR_KEY";

  const API_URL = "http://127.0.0.1:5001";


  chrome.tabs.query(
    {
      active: true,
      currentWindow: true
    },

    async (tabs) => {

      const url = tabs[0]?.url || "";

      const youtubeRegex =
        /(?:youtube\.com\/watch\?v=|youtu\.be\/)([\w-]{11})/;

      const match = url.match(youtubeRegex);


      if (!match || !match[1]) {

        outputDiv.innerHTML =
          "<p class='error'>Open a valid YouTube video first.</p>";

        return;
      }


      const videoId = match[1];


      outputDiv.innerHTML = `
        <div class="section-title">
          YouTube Video
        </div>

        <p>Video ID: ${videoId}</p>

        <p>Fetching comments...</p>
      `;


      const comments =
        await fetchComments(videoId);


      if (!comments.length) {

        outputDiv.innerHTML +=
          "<p>No comments were found.</p>";

        return;
      }


      outputDiv.innerHTML += `
        <p>
          Fetched ${comments.length} comments.
          Running sentiment analysis...
        </p>
      `;


      const predictions =
        await getSentimentPredictions(
          comments
        );


      if (!predictions) {
        return;
      }


      const sentimentCounts = {
        "1": 0,
        "0": 0,
        "-1": 0
      };


      const sentimentData = [];


      predictions.forEach(item => {

        const sentiment =
          String(item.sentiment);

        if (
          sentimentCounts[sentiment]
          !== undefined
        ) {
          sentimentCounts[sentiment]++;
        }

        sentimentData.push({
          timestamp: item.timestamp,
          sentiment: Number(item.sentiment)
        });

      });


      const totalComments =
        comments.length;


      const uniqueCommenters =
        new Set(
          comments.map(
            comment => comment.authorId
          )
        ).size;


      const totalWords =
        comments.reduce(
          (sum, comment) => {

            const words =
              comment.text
                .split(/\s+/)
                .filter(
                  word => word.length > 0
                );

            return sum + words.length;

          },
          0
        );


      const avgCommentLength =
        (
          totalWords /
          totalComments
        ).toFixed(2);


      const totalSentimentScore =
        predictions.reduce(
          (sum, item) =>
            sum + Number(item.sentiment),
          0
        );


      const avgSentimentScore =
        totalSentimentScore /
        totalComments;


      const normalizedSentimentScore =
        (
          (
            (avgSentimentScore + 1) /
            2
          ) * 10
        ).toFixed(2);


      outputDiv.innerHTML += `

        <div class="section">

          <div class="section-title">
            Comment Analysis Summary
          </div>

          <div class="metrics-container">

            <div class="metric">

              <div class="metric-title">
                Total Comments
              </div>

              <div class="metric-value">
                ${totalComments}
              </div>

            </div>


            <div class="metric">

              <div class="metric-title">
                Unique Commenters
              </div>

              <div class="metric-value">
                ${uniqueCommenters}
              </div>

            </div>


            <div class="metric">

              <div class="metric-title">
                Avg Comment Length
              </div>

              <div class="metric-value">
                ${avgCommentLength} words
              </div>

            </div>


            <div class="metric">

              <div class="metric-title">
                Sentiment Score
              </div>

              <div class="metric-value">
                ${normalizedSentimentScore}/10
              </div>

            </div>

          </div>

        </div>


        <div class="section">

          <div class="section-title">
            Sentiment Distribution
          </div>

          <div id="chart-container"></div>

        </div>


        <div class="section">

          <div class="section-title">
            Sentiment Trend
          </div>

          <div id="trend-container"></div>

        </div>


        <div class="section">

          <div class="section-title">
            Comment Word Cloud
          </div>

          <div id="wordcloud-container"></div>

        </div>

      `;


      await fetchAndDisplayChart(
        sentimentCounts
      );


      await fetchAndDisplayTrendGraph(
        sentimentData
      );


      await fetchAndDisplayWordCloud(
        comments.map(
          comment => comment.text
        )
      );


      outputDiv.innerHTML += `

        <div class="section">

          <div class="section-title">
            Top 25 Comments
          </div>

          <ul class="comment-list">

            ${
              predictions
                .slice(0, 25)
                .map(
                  (item, index) => `

                    <li class="comment-item">

                      <span>
                        ${index + 1}.
                        ${item.comment}
                      </span>

                      <br>

                      <span class="comment-sentiment">
                        Sentiment:
                        ${getSentimentLabel(
                          item.sentiment
                        )}
                      </span>

                    </li>

                  `
                )
                .join("")
            }

          </ul>

        </div>
      `;

    }
  );


  // ------------------------------------------------
  // Fetch YouTube comments
  // ------------------------------------------------

  async function fetchComments(videoId) {

    const comments = [];

    let pageToken = "";


    try {

      while (comments.length < 500) {

        const params =
          new URLSearchParams({

            part: "snippet",

            videoId: videoId,

            maxResults: "100",

            key: API_KEY

          });


        if (pageToken) {
          params.set(
            "pageToken",
            pageToken
          );
        }


        const response =
          await fetch(

            `https://www.googleapis.com/youtube/v3/commentThreads?${params.toString()}`

          );


        const data =
          await response.json();


        if (!response.ok) {

          throw new Error(
            data?.error?.message ||
            "YouTube API request failed"
          );

        }


        if (data.items) {

          data.items.forEach(item => {

            const snippet =
              item
                .snippet
                .topLevelComment
                .snippet;


            comments.push({

              text:
                snippet.textOriginal,

              timestamp:
                snippet.publishedAt,

              authorId:
                snippet
                  .authorChannelId
                  ?.value ||
                "Unknown"

            });

          });

        }


        pageToken =
          data.nextPageToken;


        if (!pageToken) {
          break;
        }

      }

    }
    catch (error) {

      console.error(
        "YouTube comments error:",
        error
      );

      outputDiv.innerHTML += `
        <p class="error">
          Error fetching YouTube comments:
          ${error.message}
        </p>
      `;

    }


    return comments;
  }


  // ------------------------------------------------
  // Prediction
  // ------------------------------------------------

  async function getSentimentPredictions(
    comments
  ) {

    try {

      const response =
        await fetch(
          `${API_URL}/predict_with_timestamps`,
          {

            method: "POST",

            headers: {
              "Content-Type":
                "application/json"
            },

            body:
              JSON.stringify({
                comments
              })

          }
        );


      const result =
        await response.json();


      if (!response.ok) {

        throw new Error(
          result.error ||
          "Prediction request failed"
        );

      }


      return result;

    }
    catch (error) {

      console.error(
        "Prediction error:",
        error
      );


      outputDiv.innerHTML += `
        <p class="error">
          Error running sentiment analysis:
          ${error.message}
        </p>
      `;


      return null;

    }
  }


  // ------------------------------------------------
  // Pie chart
  // ------------------------------------------------

  async function fetchAndDisplayChart(
    sentimentCounts
  ) {

    try {

      const response =
        await fetch(
          `${API_URL}/generate_chart`,
          {

            method: "POST",

            headers: {
              "Content-Type":
                "application/json"
            },

            body:
              JSON.stringify({
                sentiment_counts:
                  sentimentCounts
              })

          }
        );


      if (!response.ok) {

        throw new Error(
          "Failed to generate chart"
        );

      }


      const blob =
        await response.blob();


      displayImage(
        blob,
        "chart-container"
      );

    }
    catch (error) {

      console.error(
        "Chart error:",
        error
      );

    }
  }


  // ------------------------------------------------
  // Word cloud
  // ------------------------------------------------

  async function fetchAndDisplayWordCloud(
    comments
  ) {

    try {

      const response =
        await fetch(
          `${API_URL}/generate_wordcloud`,
          {

            method: "POST",

            headers: {
              "Content-Type":
                "application/json"
            },

            body:
              JSON.stringify({
                comments
              })

          }
        );


      if (!response.ok) {

        throw new Error(
          "Failed to generate word cloud"
        );

      }


      const blob =
        await response.blob();


      displayImage(
        blob,
        "wordcloud-container"
      );

    }
    catch (error) {

      console.error(
        "Word cloud error:",
        error
      );

    }
  }


  // ------------------------------------------------
  // Trend graph
  // ------------------------------------------------

  async function fetchAndDisplayTrendGraph(
    sentimentData
  ) {

    try {

      const response =
        await fetch(
          `${API_URL}/generate_trend_graph`,
          {

            method: "POST",

            headers: {
              "Content-Type":
                "application/json"
            },

            body:
              JSON.stringify({
                sentiment_data:
                  sentimentData
              })

          }
        );


      if (!response.ok) {

        throw new Error(
          "Failed to generate trend graph"
        );

      }


      const blob =
        await response.blob();


      displayImage(
        blob,
        "trend-container"
      );

    }
    catch (error) {

      console.error(
        "Trend graph error:",
        error
      );

    }
  }


  // ------------------------------------------------
  // Image helper
  // ------------------------------------------------

  function displayImage(
    blob,
    containerId
  ) {

    const imgURL =
      URL.createObjectURL(blob);


    const img =
      document.createElement("img");


    img.src = imgURL;


    const container =
      document.getElementById(
        containerId
      );


    if (container) {
      container.appendChild(img);
    }

  }


  // ------------------------------------------------
  // Sentiment labels
  // ------------------------------------------------

  function getSentimentLabel(
    sentiment
  ) {

    const value =
      Number(sentiment);


    if (value === 1) {
      return "Positive";
    }

    if (value === -1) {
      return "Negative";
    }

    return "Neutral";
  }

});