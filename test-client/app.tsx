import React, { useState, useEffect, useRef, useMemo } from 'react';
import { createRoot } from 'react-dom/client';

import {
  Grid, Icon, VStack, Button, ChakraProvider,
  Box, FormControl, FormLabel, Select, Center, Alert, AlertIcon, Progress, useColorMode, IconButton, GridItem, Table, Thead, Tr, Td, Tbody, HStack, Drawer, Heading, DrawerOverlay, DrawerContent, DrawerCloseButton, DrawerHeader, DrawerBody, Container, Text
} from '@chakra-ui/react';
import { FaImage, FaInfoCircle, FaMoon, FaStop, FaSun } from 'react-icons/fa';

createRoot(document.getElementById('root'))
  .render(
    <ChakraProvider>
      <App />
    </ChakraProvider>
  );

const WIDTH = 600;
const HEIGHT = 450;
const SERVER = "ws://localhost:8080";

interface RenderResult {
  imageBitmap: ImageBitmap,
  /** In seconds with millisecond precision. */
  timeToRender: number,
}

interface RenderJob {
  start: number,
  pixelsRendered: number,
}

const options = [
  ["Cornell Box", "cornell_box"],
  ["Cubes", "cubes"],
  ["Flying Unicorn", "flying_unicorn"],
];

let renderJob: RenderJob | null = null;
let connection: Connection;

enum MessageType {
  RenderedPixels = 0,
}

function App() {
  const [scene, setScene] = useState('cornell_box')
  const [renderResults, setRenderResults] = useState([] as RenderResult[]);
  const { toggleColorMode, colorMode } = useColorMode();
  const canvasRef = useRef(null as HTMLCanvasElement | null);
  const [error, setError] = useState(null);

  const onMessage = async (e) => {
    const view = new DataView(e.data);
    const messageType = view.getUint8(0) as MessageType;
    switch (messageType) {
    case MessageType.RenderedPixels:
      const numPixels = view.getUint8(1);
      const x = view.getUint16(2, true);
      const y = view.getUint16(4, true);
      const data = new Uint8ClampedArray(view.buffer, 6);
      const imageData = new ImageData(numPixels, 1);
      for (let i = 0; i < numPixels; i++) {
        imageData.data.set(data.subarray(i * 3, i * 3 + 3), i * 4);
        imageData.data[i * 4 + 3] = 255;
      }
      const ctx = canvasRef.current?.getContext('2d')!;
      ctx.putImageData(imageData, x, y);

      renderJob!.pixelsRendered += numPixels;
      if (renderJob!.pixelsRendered >= WIDTH * HEIGHT) {
        const timeToRender = (Date.now() - renderJob!.start) / 1000;
        const imageBitmap =
          await createImageBitmap(ctx.getImageData(0, 0, WIDTH, HEIGHT));
        setRenderResults(rrs =>
          [{ imageBitmap, timeToRender }, ...rrs]
        );
        renderJob = null;
      }
      break;
    }
  };

  useEffect(() => {
    const ctx = canvasRef.current!.getContext('2d')!;
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, WIDTH, HEIGHT);

    connection = new Conn
  }, []);

  return (
    <Grid
      w='100vw'
      h='100vh'
      templateAreas={`
        "canvas controls"
        "canvas results"
      `}
      gridTemplateColumns='700px 1fr'
      gridTemplateRows='50vh 50vh'
    >
      <HStack position='absolute' top={4} left={4} spacing={4}>
        <About/>
        <ColorModeToggle isDark={colorMode === 'light'} onClick={toggleColorMode} />
      </HStack>
      <GridItem>
        <Center h='100vh'>
          <VStack spacing={25}>
            <canvas
              ref={canvasRef}
              width={WIDTH}
              height={HEIGHT}
              style={{
                boxShadow:
                  colorMode === 'light' ?
                    '0px 8px 15px #999' :
                    'none'
              }}
            />
            {Boolean(renderJob) ?
              <Progress
                min={0}
                max={WIDTH * HEIGHT}
                value={renderJob!.pixelsRendered}
                size="lg"
                // style={{ width: '100%' }}
              /> :
              null
            }
          </VStack>
        </Center>
      </GridItem>
      <GridItem m='20px'>
        <Box
          as='form'
          onSubmit={e => {
            e.preventDefault();
            const request = 
              Boolean(renderJob) ?
                { type: 'stop_rendering' } :
                { type: 'render', scene };
            renderJob = {
              pixelsRendered: 0,
              start: Date.now(),
            };
            connection.send(JSON.stringify(request));
          }}
        >
          <VStack>
            <FormControl>
              <FormLabel>Scene</FormLabel>
              <Select
                value={scene}
                onChange={e => setScene(e.target.value)}
              >
                {options.map(([name, value], i) => (
                  <option key={`opt${i}`} value={value}>{name}</option>
                ))}
              </Select>
            </FormControl>
            <Button type="submit" leftIcon={Boolean(renderJob) ? <FaStop /> : undefined}>
              {Boolean(renderJob) ? <>Stop</> : 'Render'}
            </Button>
            {error !== null ?
              <Alert status='error' variant='left-accent'>
                <AlertIcon/>
                Could not connect to the server.
              </Alert> :
              null
            }
          </VStack>
        </Box>
      </GridItem>
      <GridItem m='20px' colStart={2}>
        <Table>
          <Thead>
            <Tr>
              <Td w={120}><Icon as={FaImage}/></Td>
              <Td>Time to Render</Td>
            </Tr>
          </Thead>
          <Tbody>
            {renderResults.map(({ imageBitmap, timeToRender }, i) => (
              <Tr key={`row${i}`}>
                <Td>
                  <ImageBitmapView bitmap={imageBitmap}/>
                </Td>
                <Td>{timeToRender.toFixed(1)}s</Td>
              </Tr>
            ))}
          </Tbody>
        </Table>
      </GridItem>
    </Grid>
  )
}

// Serialization format for rendered pixels:
//   MsgLength = 2 + 8 * NumPixels
//
//   HEADER (2 bytes)
//        [0]  Message Type (u8, always 0)
//        [1]  Num Records (u8)
//
//   PIXEL i (8 bytes)
//     [3i+6]  r (u8)
//     [3i+7]  g (u8)
//     [3i+8]  b (u8)
function deserializeRenderedPixels(view: DataView): ImageData {
  return imageData;
}


function About() {
  const [isOpen, setIsOpen] = useState(false);
  const btnRef = useRef(null as HTMLButtonElement | null);

  return (
    <>
      <IconButton 
        ref={btnRef}
        variant='solid'
        aria-label='Color mode toggle'
        rounded='full'
        size='sm'
        icon={<FaInfoCircle/>}
        onClick={() => {
          setIsOpen(true);
        }}
      />
      <Drawer
        isOpen={isOpen}
        placement='left'
        size='full'
        onClose={() => {
          setIsOpen(false);
        }}
        finalFocusRef={btnRef}
      >
        <DrawerOverlay/>
        <DrawerContent>
          <DrawerCloseButton
            position='relative'
            top={2}
            left={4}
          />
          <DrawerBody>
            <Container>
              <VStack spacing={4} align='left'>
                <Heading size='3xl'>About</Heading>
                <Text fontSize='xl'>Frontend for a raytracer written in Rust.</Text>
                <Heading size='lg'>How it Works</Heading>
                {/* <Canvas
                  maxWidth={800}
                  maxHeight={600}
                  nodes={[
                    {
                      id: '1',
                      text: '1'
                    },
                    {
                      id: '2',
                      text: '2'
                    }
                  ]}
                  edges={[
                    {
                      id: '1-2',
                      from: '1',
                      to: '2'
                    }
                  ]}
                /> */}
              </VStack>
            </Container>
          </DrawerBody>
        </DrawerContent>
      </Drawer>
    </>
  )
}

function ColorModeToggle({ isDark, onClick }) {
  return (
    <IconButton
      variant='solid'
      aria-label='Color mode toggle'
      rounded='full'
      size='sm'
      icon={<Icon as={isDark ? FaSun : FaMoon} />}
      onClick={onClick}
    />
  )
}

function ImageBitmapView({ bitmap }: {
  bitmap: ImageBitmap,
}) {
  const canvasRef = useRef(null as HTMLCanvasElement | null);
  const width = 100;
  const height = width * bitmap.height / bitmap.width;

  useEffect(() => {
    (async () => {
      const resizedBitmap = await createImageBitmap(bitmap, {
        resizeWidth: width,
        resizeHeight: height
      });
      canvasRef
        .current
        ?.getContext('bitmaprenderer')
        ?.transferFromImageBitmap(resizedBitmap);
    })()
  }, [canvasRef, bitmap]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
    />
  );
}

enum ServerMessage {
  RenderedPixels = 0,
}
