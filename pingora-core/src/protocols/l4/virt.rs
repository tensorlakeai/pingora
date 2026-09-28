//! Provides [`VirtualSocketStream`].

use std::{
    pin::Pin,
    task::{Context, Poll},
};

use tokio::io::{AsyncRead, AsyncWrite};

use super::ext::TcpKeepalive;

/// A limited set of socket options that can be set on a [`VirtualSocket`].
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum VirtualSockOpt {
    NoDelay,
    KeepAlive(TcpKeepalive),
}

/// A "virtual" socket that supports async read and write operations.
pub trait VirtualSocket: AsyncRead + AsyncWrite + Unpin + Send + Sync + std::fmt::Debug {
    /// Set a socket option.
    fn set_socket_option(&self, opt: VirtualSockOpt) -> std::io::Result<()>;

    /// The OS socket this virtual socket is layered over, if any.
    ///
    /// The connection pool only reuses a stream whose fd it can verify against
    /// the peer address, so a virtual socket that keeps the default `None`
    /// (reported as fd -1) is never reused: every request opens a new stream.
    /// A virtual socket that wraps a real socket, such as a TLS layer over a
    /// TCP connection, returns that socket's fd to opt in to pooling.
    #[cfg(unix)]
    fn as_raw_fd(&self) -> Option<std::os::unix::io::RawFd> {
        None
    }

    /// Windows counterpart of [`VirtualSocket::as_raw_fd`].
    #[cfg(windows)]
    fn as_raw_socket(&self) -> Option<std::os::windows::io::RawSocket> {
        None
    }
}

/// Wrapper around any type implementing  [`VirtualSocket`].
#[derive(Debug)]
pub struct VirtualSocketStream {
    pub(crate) socket: Box<dyn VirtualSocket>,
}

impl VirtualSocketStream {
    pub fn new(socket: Box<dyn VirtualSocket>) -> Self {
        Self { socket }
    }

    #[inline]
    pub fn set_socket_option(&self, opt: VirtualSockOpt) -> std::io::Result<()> {
        self.socket.set_socket_option(opt)
    }
}

impl AsyncRead for VirtualSocketStream {
    #[inline]
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut tokio::io::ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        Pin::new(&mut *self.get_mut().socket).poll_read(cx, buf)
    }
}

impl AsyncWrite for VirtualSocketStream {
    #[inline]
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<std::io::Result<usize>> {
        Pin::new(&mut *self.get_mut().socket).poll_write(cx, buf)
    }

    #[inline]
    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        Pin::new(&mut *self.get_mut().socket).poll_flush(cx)
    }

    #[inline]
    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        Pin::new(&mut *self.get_mut().socket).poll_shutdown(cx)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use tokio::io::{AsyncReadExt, AsyncWriteExt as _};

    use crate::protocols::l4::stream::Stream;

    use super::*;

    #[derive(Debug)]
    struct StaticVirtualSocket {
        content: Vec<u8>,
        read_pos: usize,
        write_buf: Arc<Mutex<Vec<u8>>>,
    }

    impl AsyncRead for StaticVirtualSocket {
        fn poll_read(
            mut self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            buf: &mut tokio::io::ReadBuf<'_>,
        ) -> Poll<std::io::Result<()>> {
            debug_assert!(self.read_pos <= self.content.len());

            let remaining = self.content.len() - self.read_pos;
            if remaining == 0 {
                return Poll::Ready(Ok(()));
            }

            let to_read = std::cmp::min(remaining, buf.remaining());
            buf.put_slice(&self.content[self.read_pos..self.read_pos + to_read]);
            self.read_pos += to_read;

            Poll::Ready(Ok(()))
        }
    }

    impl AsyncWrite for StaticVirtualSocket {
        fn poll_write(
            self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            buf: &[u8],
        ) -> Poll<std::io::Result<usize>> {
            // write to internal buffer
            let this = self.get_mut();
            this.write_buf.lock().unwrap().extend_from_slice(buf);
            Poll::Ready(Ok(buf.len()))
        }

        fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
            Poll::Ready(Ok(()))
        }

        fn poll_shutdown(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
            Poll::Ready(Ok(()))
        }
    }

    impl VirtualSocket for StaticVirtualSocket {
        fn set_socket_option(&self, _opt: VirtualSockOpt) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// A virtual socket layered over a real TCP socket.
    #[derive(Debug)]
    struct TcpBackedVirtualSocket(tokio::net::TcpStream);

    impl AsyncRead for TcpBackedVirtualSocket {
        fn poll_read(
            mut self: Pin<&mut Self>,
            cx: &mut Context<'_>,
            buf: &mut tokio::io::ReadBuf<'_>,
        ) -> Poll<std::io::Result<()>> {
            Pin::new(&mut self.0).poll_read(cx, buf)
        }
    }

    impl AsyncWrite for TcpBackedVirtualSocket {
        fn poll_write(
            mut self: Pin<&mut Self>,
            cx: &mut Context<'_>,
            buf: &[u8],
        ) -> Poll<std::io::Result<usize>> {
            Pin::new(&mut self.0).poll_write(cx, buf)
        }

        fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
            Pin::new(&mut self.0).poll_flush(cx)
        }

        fn poll_shutdown(
            mut self: Pin<&mut Self>,
            cx: &mut Context<'_>,
        ) -> Poll<std::io::Result<()>> {
            Pin::new(&mut self.0).poll_shutdown(cx)
        }
    }

    impl VirtualSocket for TcpBackedVirtualSocket {
        fn set_socket_option(&self, _opt: VirtualSockOpt) -> std::io::Result<()> {
            Ok(())
        }

        #[cfg(unix)]
        fn as_raw_fd(&self) -> Option<std::os::unix::io::RawFd> {
            use std::os::unix::io::AsRawFd;
            Some(self.0.as_raw_fd())
        }
    }

    /// A virtual socket that does not expose an fd keeps the historical
    /// `-1` id, so the connection pool never reuses it.
    #[cfg(unix)]
    #[tokio::test]
    async fn test_virtual_socket_without_fd_reports_minus_one() {
        use crate::protocols::UniqueID;

        let socket = StaticVirtualSocket {
            content: Vec::new(),
            read_pos: 0,
            write_buf: Arc::new(Mutex::new(Vec::new())),
        };
        let stream: Stream = VirtualSocketStream::new(Box::new(socket)).into();
        assert_eq!(stream.id(), -1);
    }

    /// A virtual socket layered over a TCP socket reports that socket's fd,
    /// which is what lets the connection pool verify and reuse it.
    #[cfg(unix)]
    #[tokio::test]
    async fn test_virtual_socket_with_fd_reports_underlying_fd() {
        use crate::protocols::UniqueID;
        use std::os::unix::io::AsRawFd;

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let accept = tokio::spawn(async move { listener.accept().await.unwrap() });
        let tcp = tokio::net::TcpStream::connect(addr).await.unwrap();
        let _server_side = accept.await.unwrap();
        let fd = tcp.as_raw_fd();

        let stream: Stream = VirtualSocketStream::new(Box::new(TcpBackedVirtualSocket(tcp))).into();
        assert_eq!(stream.id(), fd);
    }

    /// Basic test that ensures reading and writing works with a virtual socket.
    //
    /// Mostly just ensures that construction works and the plumbing is correct.
    #[tokio::test]
    async fn test_stream_virtual() {
        let content = b"hello virtual world";
        let write_buf = Arc::new(Mutex::new(Vec::new()));
        let mut stream = Stream::from(VirtualSocketStream::new(Box::new(StaticVirtualSocket {
            content: content.to_vec(),
            read_pos: 0,
            write_buf: write_buf.clone(),
        })));

        let mut buf = Vec::new();
        let out = stream.read_to_end(&mut buf).await.unwrap();
        assert_eq!(out, content.len());
        assert_eq!(buf, content);

        stream.write_all(content).await.unwrap();
        assert_eq!(write_buf.lock().unwrap().as_slice(), content);
    }
}
